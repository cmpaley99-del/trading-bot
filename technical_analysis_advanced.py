"""
Advanced Technical Analysis with Pattern Recognition
Extends basic technical analysis with harmonic patterns, Elliott waves, and Fibonacci analysis
"""

import pandas as pd
import numpy as np
from loguru import logger
from technical_analysis import technical_analysis
from pattern_recognition import pattern_recognition


class AdvancedTechnicalAnalysis:
    """Advanced technical analysis combining traditional indicators with pattern recognition"""

    def __init__(self):
        self.basic_ta = technical_analysis
        logger.info("Advanced Technical Analysis initialized")

    def generate_advanced_signals(self, df):
        """Generate comprehensive trading signals including advanced patterns"""
        if df is None or len(df) < 50:
            logger.warning("Insufficient data for advanced technical analysis")
            return None

        try:
            # Get basic technical signals
            basic_signals = self.basic_ta.generate_signals(df)

            if basic_signals is None:
                return None

            # Get advanced pattern signals
            pattern_signals = self._get_advanced_pattern_signals(df)

            # Combine all signals
            combined_signals = {**basic_signals, **pattern_signals}

            # Re-evaluate overall signal with advanced patterns
            combined_signals['overall_signal'] = self._determine_advanced_overall_signal(combined_signals)

            # Add pattern confidence to signal strength
            combined_signals['signal_strength'] = self._calculate_signal_strength(combined_signals)

            return combined_signals

        except Exception as e:
            logger.error(f"Error in advanced signal generation: {e}")
            return None

    def _get_advanced_pattern_signals(self, df):
        """Get advanced pattern recognition signals"""
        try:
            # Get pattern analysis from the pattern recognition module
            pattern_analysis = pattern_recognition.get_pattern_signals(df)

            return {
                'harmonic_signal': pattern_analysis.get('overall_signal', 'NEUTRAL'),
                'elliott_signal': self._convert_elliott_to_signal(pattern_analysis.get('elliott_position', '')),
                'fibonacci_signal': pattern_analysis.get('fibonacci_signals', {}).get('signal', 'NEUTRAL'),
                'pattern_confidence': pattern_analysis.get('confidence', 0.0),
                'harmonic_patterns': pattern_analysis.get('harmonic_patterns', {}),
                'elliott_position': pattern_analysis.get('elliott_position', ''),
                'fibonacci_level': pattern_analysis.get('fibonacci_level', {})
            }
        except Exception as e:
            logger.warning(f"Error getting advanced pattern signals: {e}")
            return {
                'harmonic_signal': 'NEUTRAL',
                'elliott_signal': 'NEUTRAL',
                'fibonacci_signal': 'NEUTRAL',
                'pattern_confidence': 0.0,
                'harmonic_patterns': {},
                'elliott_position': '',
                'fibonacci_level': {}
            }

    def _convert_elliott_to_signal(self, position):
        """Convert Elliott wave position to trading signal"""
        if not position:
            return 'NEUTRAL'

        position = position.lower()

        if 'wave_5_completion' in position:
            return 'BEARISH'  # Wave 5 completion often signals reversal
        elif 'wave_c_completion' in position:
            return 'BULLISH'  # Wave C completion often signals reversal
        elif 'wave_5' in position:
            return 'NEUTRAL'  # In wave 5, follow trend
        elif 'transition' in position:
            return 'NEUTRAL'  # Transition periods are uncertain
        else:
            return 'NEUTRAL'

    def _determine_advanced_overall_signal(self, signals):
        """Determine overall signal incorporating advanced patterns"""
        signal_weights = {
            'rsi_signal': 1.0,
            'macd_signal': 1.2,
            'bb_signal': 0.8,
            'trend_signal': 1.5,
            'volume_signal': 1.0,
            'pattern_signal': 0.8,
            'ema_scalp_signal': 0.7,
            'momentum_signal': 1.0,
            'vwap_signal': 0.9,
            'harmonic_signal': 2.0,  # Higher weight for advanced patterns
            'elliott_signal': 1.8,
            'fibonacci_signal': 1.5
        }

        bullish_score = 0
        bearish_score = 0

        for signal_name, signal_value in signals.items():
            if signal_name in signal_weights and signal_value in ['BULLISH', 'BEARISH']:
                weight = signal_weights.get(signal_name, 1.0)
                if signal_value == 'BULLISH':
                    bullish_score += weight
                else:
                    bearish_score += weight

        # Apply pattern confidence multiplier
        pattern_confidence = signals.get('pattern_confidence', 0.0)
        if pattern_confidence > 0.7:
            # Boost the winning side if pattern confidence is high
            if bullish_score > bearish_score:
                bullish_score *= (1 + pattern_confidence)
            elif bearish_score > bullish_score:
                bearish_score *= (1 + pattern_confidence)

        # Determine final signal
        if bullish_score >= bearish_score * 1.3:  # 30% advantage needed
            return 'BULLISH'
        elif bearish_score >= bullish_score * 1.3:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _calculate_signal_strength(self, signals):
        """Calculate overall signal strength based on confluence"""
        signal_types = [
            'rsi_signal', 'macd_signal', 'bb_signal', 'trend_signal',
            'volume_signal', 'pattern_signal', 'harmonic_signal',
            'elliott_signal', 'fibonacci_signal'
        ]

        bullish_count = sum(1 for sig in signal_types
                          if signals.get(sig) == 'BULLISH')
        bearish_count = sum(1 for sig in signal_types
                          if signals.get(sig) == 'BEARISH')

        total_signals = len(signal_types)
        max_confluence = max(bullish_count, bearish_count)

        # Base strength on confluence
        base_strength = max_confluence / total_signals

        # Boost strength with pattern confidence
        pattern_confidence = signals.get('pattern_confidence', 0.0)
        final_strength = base_strength * (1 + pattern_confidence * 0.5)

        return min(final_strength, 1.0)  # Cap at 1.0

    def get_pattern_analysis_summary(self, df):
        """Get detailed pattern analysis summary"""
        try:
            pattern_analysis = pattern_recognition.get_pattern_signals(df)

            summary = {
                'harmonic_patterns_detected': len(pattern_analysis.get('harmonic_patterns', {})),
                'elliott_wave_position': pattern_analysis.get('elliott_position', 'unknown'),
                'fibonacci_signal': pattern_analysis.get('fibonacci_signals', {}).get('signal', 'NEUTRAL'),
                'pattern_confidence': pattern_analysis.get('confidence', 0.0),
                'recommended_action': pattern_analysis.get('overall_signal', 'NEUTRAL')
            }

            return summary

        except Exception as e:
            logger.error(f"Error getting pattern analysis summary: {e}")
            return {
                'harmonic_patterns_detected': 0,
                'elliott_wave_position': 'unknown',
                'fibonacci_signal': 'NEUTRAL',
                'pattern_confidence': 0.0,
                'recommended_action': 'NEUTRAL'
            }


# Singleton instance
advanced_technical_analysis = AdvancedTechnicalAnalysis()
