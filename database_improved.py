"""
Improved Database Module
Enhanced with connection pooling, better error handling, and performance optimizations
"""

import sqlite3
import pandas as pd
import threading
from datetime import datetime, date
from typing import Optional, Dict, List, Any, Union
from contextlib import contextmanager
from functools import lru_cache
import time
from loguru import logger
from config import Config

class DatabaseConnectionPool:
    """Thread-safe database connection pool"""

    def __init__(self, db_path: str, max_connections: int = 5):
        self.db_path = db_path
        self.max_connections = max_connections
        self.connections = []
        self.lock = threading.Lock()
        self._local = threading.local()

    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection from the pool"""
        with self.lock:
            # Try to get existing connection for this thread
            if hasattr(self._local, 'connection'):
                conn = self._local.connection
                if conn:
                    try:
                        conn.execute("SELECT 1")  # Test connection
                        return conn
                    except sqlite3.Error:
                        # Connection is dead, remove it
                        self.connections.remove(conn)
                        conn.close()

            # Get connection from pool or create new one
            if self.connections:
                conn = self.connections.pop()
            else:
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row  # Enable column access by name

            # Store in thread local storage
            self._local.connection = conn
            return conn

    def return_connection(self, conn: sqlite3.Connection):
        """Return connection to pool"""
        with self.lock:
            if len(self.connections) < self.max_connections:
                try:
                    conn.rollback()  # Reset transaction state
                    self.connections.append(conn)
                except sqlite3.Error:
                    # Connection is bad, don't return to pool
                    conn.close()
            else:
                conn.close()

    def close_all(self):
        """Close all connections in the pool"""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except:
                    pass
            self.connections.clear()

class ImprovedDatabase:
    """Improved database class with connection pooling and error handling"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or Config.DATABASE_PATH
        self.connection_pool = DatabaseConnectionPool(self.db_path)
        self._initialized = False
        self._init_lock = threading.Lock()

        # Initialize database tables
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Ensure database is initialized (thread-safe)"""
        if self._initialized:
            return

        with self._init_lock:
            if not self._initialized:
                self._create_tables()
                self._create_indexes()
                self._initialized = True

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = self.connection_pool.get_connection()
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                try:
                    conn.rollback()
                except:
                    pass
            raise
        finally:
            if conn:
                self.connection_pool.return_connection(conn)

    def _create_tables(self):
        """Create all necessary database tables"""
        with self.get_connection() as conn:
            try:
                # Signals table with improved schema
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        entry_price REAL,
                        position_size REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        leverage INTEGER,
                        risk_percentage REAL,
                        signal_quality TEXT,
                        confidence_score REAL,
                        rsi_signal TEXT,
                        macd_signal TEXT,
                        bb_signal TEXT,
                        trend_signal TEXT,
                        volume_signal TEXT,
                        pattern_signal TEXT,
                        harmonic_pattern TEXT,
                        elliott_wave TEXT,
                        fibonacci_level REAL,
                        message TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Trades table with improved tracking
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        signal_id INTEGER,
                        entry_time DATETIME,
                        exit_time DATETIME,
                        entry_price REAL,
                        exit_price REAL,
                        position_size REAL,
                        pnl REAL,
                        pnl_percentage REAL,
                        status TEXT DEFAULT 'OPEN',
                        stop_loss_hit BOOLEAN DEFAULT FALSE,
                        take_profit_hit BOOLEAN DEFAULT FALSE,
                        exit_reason TEXT,
                        fees REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (signal_id) REFERENCES signals (id)
                    )
                ''')

                # Performance metrics table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date DATE UNIQUE,
                        total_trades INTEGER DEFAULT 0,
                        winning_trades INTEGER DEFAULT 0,
                        losing_trades INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0,
                        total_pnl REAL DEFAULT 0,
                        average_pnl REAL DEFAULT 0,
                        max_drawdown REAL DEFAULT 0,
                        sharpe_ratio REAL DEFAULT 0,
                        total_volume REAL DEFAULT 0,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Metadata table for bot state and configuration
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        value_type TEXT DEFAULT 'string',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')

                # Anomaly logs table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS anomaly_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        symbol TEXT,
                        anomaly_type TEXT,
                        severity TEXT,
                        description TEXT,
                        price_at_detection REAL,
                        volume_at_detection REAL,
                        confidence_score REAL,
                        action_taken TEXT
                    )
                ''')

                conn.commit()
                logger.info("✅ Database tables created successfully")

            except Exception as e:
                logger.error(f"❌ Error creating database tables: {e}")
                raise

    def _create_indexes(self):
        """Create database indexes for better performance"""
        with self.get_connection() as conn:
            try:
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)",
                    "CREATE INDEX IF NOT EXISTS idx_signals_type ON signals(signal_type)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_signal_id ON trades(signal_id)",
                    "CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)",
                    "CREATE INDEX IF NOT EXISTS idx_performance_date ON performance(date)",
                    "CREATE INDEX IF NOT EXISTS idx_anomaly_logs_timestamp ON anomaly_logs(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_anomaly_logs_symbol ON anomaly_logs(symbol)"
                ]

                for index_sql in indexes:
                    conn.execute(index_sql)

                conn.commit()
                logger.info("✅ Database indexes created successfully")

            except Exception as e:
                logger.error(f"❌ Error creating database indexes: {e}")

    def save_signal(self, signal_data: Dict[str, Any]) -> Optional[int]:
        """Save a trading signal to the database with improved error handling"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals (
                        symbol, signal_type, entry_price, position_size, stop_loss,
                        take_profit, leverage, risk_percentage, signal_quality,
                        confidence_score, rsi_signal, macd_signal, bb_signal,
                        trend_signal, volume_signal, pattern_signal,
                        harmonic_pattern, elliott_wave, fibonacci_level, message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data.get('symbol', Config.TRADING_PAIR),
                    signal_data.get('signal_type', 'UNKNOWN'),
                    signal_data.get('entry_price'),
                    signal_data.get('position_size'),
                    signal_data.get('stop_loss'),
                    signal_data.get('take_profit'),
                    signal_data.get('leverage', Config.DEFAULT_LEVERAGE),
                    signal_data.get('risk_percentage', Config.RISK_PERCENTAGE),
                    signal_data.get('signal_quality', 'UNKNOWN'),
                    signal_data.get('confidence_score', 0.0),
                    signal_data.get('rsi_signal'),
                    signal_data.get('macd_signal'),
                    signal_data.get('bb_signal'),
                    signal_data.get('trend_signal'),
                    signal_data.get('volume_signal'),
                    signal_data.get('pattern_signal'),
                    signal_data.get('harmonic_pattern'),
                    signal_data.get('elliott_wave'),
                    signal_data.get('fibonacci_level'),
                    signal_data.get('message')
                ))

                conn.commit()
                signal_id = cursor.lastrowid
                logger.info(f"✅ Signal saved with ID: {signal_id}")
                return signal_id

        except Exception as e:
            logger.error(f"❌ Error saving signal: {e}")
            return None

    def save_trade(self, trade_data: Dict[str, Any]) -> Optional[int]:
        """Save a completed trade to the database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trades (
                        signal_id, entry_time, exit_time, entry_price, exit_price,
                        position_size, pnl, pnl_percentage, status,
                        stop_loss_hit, take_profit_hit, exit_reason, fees
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_data.get('signal_id'),
                    trade_data.get('entry_time'),
                    trade_data.get('exit_time'),
                    trade_data.get('entry_price'),
                    trade_data.get('exit_price'),
                    trade_data.get('position_size'),
                    trade_data.get('pnl'),
                    trade_data.get('pnl_percentage'),
                    trade_data.get('status', 'COMPLETED'),
                    trade_data.get('stop_loss_hit', False),
                    trade_data.get('take_profit_hit', False),
                    trade_data.get('exit_reason'),
                    trade_data.get('fees', 0)
                ))

                conn.commit()
                trade_id = cursor.lastrowid
                logger.info(f"✅ Trade saved with ID: {trade_id}")
                return trade_id

        except Exception as e:
            logger.error(f"❌ Error saving trade: {e}")
            return None

    def get_recent_signals(self, limit: int = 10, symbol: str = None) -> Optional[pd.DataFrame]:
        """Get recent trading signals with optional symbol filter"""
        try:
            with self.get_connection() as conn:
                query = '''
                    SELECT * FROM signals
                    WHERE 1=1
                '''
                params = []

                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)

                query += f' ORDER BY timestamp DESC LIMIT {limit}'

                df = pd.read_sql_query(query, conn, params=params)
                return df

        except Exception as e:
            logger.error(f"❌ Error getting recent signals: {e}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        try:
            with self.get_connection() as conn:
                # Get trade statistics
                trades_stats = pd.read_sql_query('''
                    SELECT
                        COUNT(*) as total_trades,
                        SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                        SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
                        AVG(pnl) as average_pnl,
                        SUM(pnl) as total_pnl,
                        MIN(pnl) as worst_trade,
                        MAX(pnl) as best_trade,
                        AVG(pnl_percentage) as average_pnl_percentage
                    FROM trades
                    WHERE status = 'COMPLETED'
                ''', conn)

                if trades_stats.empty:
                    return {}

                stats = trades_stats.iloc[0].to_dict()

                # Calculate additional metrics
                if stats['total_trades'] > 0:
                    stats['win_rate'] = (stats['winning_trades'] / stats['total_trades']) * 100
                    stats['profit_factor'] = (
                        abs(stats.get('total_pnl', 0)) /
                        max(abs(stats.get('total_losses', 1)), 1)
                    ) if stats.get('total_pnl', 0) > 0 else 0

                return stats

        except Exception as e:
            logger.error(f"❌ Error getting performance stats: {e}")
            return {}

    @lru_cache(maxsize=32)
    def get_signal_stats(self, days: int = 30) -> Optional[pd.DataFrame]:
        """Get signal statistics with caching"""
        try:
            with self.get_connection() as conn:
                query = f'''
                    SELECT
                        signal_type,
                        COUNT(*) as count,
                        AVG(CASE WHEN signal_quality = 'HIGH' THEN 1 ELSE 0 END) as high_quality_rate,
                        AVG(confidence_score) as avg_confidence,
                        MIN(timestamp) as first_signal,
                        MAX(timestamp) as last_signal
                    FROM signals
                    WHERE timestamp >= datetime('now', '-{days} days')
                    GROUP BY signal_type
                    ORDER BY count DESC
                '''

                df = pd.read_sql_query(query, conn)
                return df

        except Exception as e:
            logger.error(f"❌ Error getting signal stats: {e}")
            return None

    def save_last_signal_time(self, timestamp: datetime) -> bool:
        """Save the last signal timestamp with improved error handling"""
        try:
            with self.get_connection() as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO metadata (key, value, value_type, updated_at)
                    VALUES (?, ?, 'datetime', CURRENT_TIMESTAMP)
                ''', ('last_signal_time', timestamp.isoformat()))

                conn.commit()
                logger.info(f"✅ Saved last signal time: {timestamp}")
                return True

        except Exception as e:
            logger.error(f"❌ Error saving last signal time: {e}")
            return False

    def get_last_signal_time(self) -> Optional[datetime]:
        """Retrieve the last signal timestamp"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute('''
                    SELECT value FROM metadata WHERE key = ?
                ''', ('last_signal_time',))

                row = cursor.fetchone()
                if row:
                    return datetime.fromisoformat(row[0])
                return None

        except Exception as e:
            logger.error(f"❌ Error retrieving last signal time: {e}")
            return None

    def save_anomaly(self, anomaly_data: Dict[str, Any]) -> Optional[int]:
        """Save anomaly detection data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO anomaly_logs (
                        symbol, anomaly_type, severity, description,
                        price_at_detection, volume_at_detection,
                        confidence_score, action_taken
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    anomaly_data.get('symbol'),
                    anomaly_data.get('anomaly_type'),
                    anomaly_data.get('severity', 'MEDIUM'),
                    anomaly_data.get('description'),
                    anomaly_data.get('price_at_detection'),
                    anomaly_data.get('volume_at_detection'),
                    anomaly_data.get('confidence_score', 0.0),
                    anomaly_data.get('action_taken')
                ))

                conn.commit()
                anomaly_id = cursor.lastrowid
                logger.info(f"✅ Anomaly saved with ID: {anomaly_id}")
                return anomaly_id

        except Exception as e:
            logger.error(f"❌ Error saving anomaly: {e}")
            return None

    def get_recent_anomalies(self, limit: int = 20) -> Optional[pd.DataFrame]:
        """Get recent anomaly detections"""
        try:
            with self.get_connection() as conn:
                df = pd.read_sql_query(
                    f'SELECT * FROM anomaly_logs ORDER BY timestamp DESC LIMIT {limit}',
                    conn
                )
                return df

        except Exception as e:
            logger.error(f"❌ Error getting recent anomalies: {e}")
            return None

    def cleanup_old_data(self, days_to_keep: int = 90) -> bool:
        """Clean up old data to maintain database performance"""
        try:
            with self.get_connection() as conn:
                # Archive old signals
                conn.execute(f'''
                    DELETE FROM signals
                    WHERE timestamp < datetime('now', '-{days_to_keep} days')
                    AND id NOT IN (SELECT signal_id FROM trades WHERE signal_id IS NOT NULL)
                ''')

                # Archive old anomaly logs
                conn.execute(f'''
                    DELETE FROM anomaly_logs
                    WHERE timestamp < datetime('now', '-{days_to_keep} days')
                ''')

                # Clean up old performance data (keep last 365 days)
                conn.execute('''
                    DELETE FROM performance
                    WHERE date < date('now', '-365 days')
                ''')

                deleted_count = conn.total_changes
                conn.commit()

                logger.info(f"✅ Cleaned up {deleted_count} old records")
                return True

        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics and health information"""
        try:
            with self.get_connection() as conn:
                stats = {}

                # Table sizes
                tables = ['signals', 'trades', 'performance', 'anomaly_logs', 'metadata']
                for table in tables:
                    cursor = conn.execute(f'SELECT COUNT(*) FROM {table}')
                    stats[f'{table}_count'] = cursor.fetchone()[0]

                # Database file size
                import os
                if os.path.exists(self.db_path):
                    stats['database_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)

                # Connection pool stats
                stats['active_connections'] = len(self.connection_pool.connections)

                return stats

        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}

    def close(self):
        """Close database connection pool"""
        self.connection_pool.close_all()
        logger.info("✅ Database connection pool closed")

    def __del__(self):
        """Destructor to ensure proper cleanup"""
        try:
            self.close()
        except:
            pass

# Global database instance
database = ImprovedDatabase()
