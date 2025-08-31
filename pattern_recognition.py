"""
Advanced Pattern Recognition Module
Implements Harmonic Patterns, Elliott Wave Analysis, and Fibonacci Retracements
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Tuple, Dict, Optional
from loguru import logger
from config import Config


class HarmonicPatterns:
    """Detect harmonic patterns like Gartley, Butterfly, Bat, Crab"""

    def __init__(self):
        self.fib_ratios = {
            'gartley': {'XA': 0.618, 'AB': 0.618, 'BC': 0.382, 'CD': 1.272},
            'butterfly': {'XA': 0.786, 'AB': 0.382, 'BC': 0.886, 'CD': 1.618},
            'bat': {'XA': 0.382, 'AB': 0.382, 'BC': 0.5, 'CD': 1.618},
            'crab': {'XA': 0.382, 'AB': 0.382, 'BC': 0.618, 'CD': 1.618}
        }
        logger.info("Harmonic Patterns detector initialized")

    def detect_patterns(self, df: pd.DataFrame) -> Dict[str, List]:
        """Detect all harmonic patterns in the data"""
        patterns = {
            'gartley': [],
            'butterfly': [],
            'bat': [],
            'crab': []
        }

        if len(df) < 50:
            return patterns

        # Find swing points
        peaks, troughs = self._find_swing_points(df)

        # Look for patterns in recent data (last 100 points)
        recent_peaks = peaks[-20:] if len(peaks) > 20 else peaks
        recent_troughs = troughs[-20:] if len(troughs) > 20 else troughs

        # Detect each pattern type
        for pattern_type in patterns.keys():
            patterns[pattern_type] = self._detect_pattern_type(
                df, recent_peaks, recent_troughs, pattern_type
            )

        return patterns

    def _find_swing_points(self, df: pd.DataFrame, distance: int = 10) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows"""
        prices = df['high'].values
        lows = df['low'].values

        # Find peaks (swing highs)
        peaks, _ = find_peaks(prices, distance=distance, prominence=np.std(prices) * 0.5)

        # Find troughs (swing lows)
        troughs, _ = find_peaks(-lows, distance=distance, prominence=np.std(lows) * 0.5)

        return peaks.tolist(), troughs.tolist()

    def _detect_pattern_type(self, df: pd.DataFrame, peaks: List[int],
                           troughs: List[int], pattern_type: str) -> List[Dict]:
        """Detect specific harmonic pattern"""
        patterns_found = []
        ratios = self.fib_ratios[pattern_type]

        # Combine and sort all swing points
        all_points = sorted(set(peaks + troughs))

        if len(all_points) < 5:
            return patterns_found

        # Look for 5-point patterns (X-A-B-C-D)
        for i in range(len(all_points) - 4):
            points = all_points[i:i+5]

            # Get price levels
            prices = [df.iloc[idx]['close'] for idx in points]

            # Check if ratios match the pattern
            if self._validate_pattern_ratios(prices, ratios):
                pattern = {
                    'type': pattern_type,
                    'points': points,
                    'prices': prices,
                    'ratios': self._calculate_actual_ratios(prices),
                    'direction': 'bullish' if prices[-1] > prices[0] else 'bearish',
                    'completion_price': self._calculate_completion_price(prices, ratios),
                    'confidence': self._calculate_pattern_confidence(prices, ratios)
                }
                patterns_found.append(pattern)

        return patterns_found

    def _validate_pattern_ratios(self, prices: List[float], target_ratios: Dict) -> bool:
        """Validate if price ratios match the harmonic pattern"""
        if len(prices) != 5:
            return False

        X, A, B, C, D = prices

        # Calculate actual ratios
        XA = abs(A - X) / abs(C - X) if C != X else 0
        AB = abs(B - A) / abs(C - A) if C != A else 0
        BC = abs(C - B) / abs(A - B) if A != B else 0
        CD = abs(D - C) / abs(B - C) if B != C else 0

        # Check if ratios are within tolerance (10%)
        tolerance = 0.1

        checks = [
            abs(XA - target_ratios['XA']) <= tolerance,
            abs(AB - target_ratios['AB']) <= tolerance,
            abs(BC - target_ratios['BC']) <= tolerance,
            abs(CD - target_ratios['CD']) <= tolerance
        ]

        return all(checks)

    def _calculate_actual_ratios(self, prices: List[float]) -> Dict[str, float]:
        """Calculate actual Fibonacci ratios for the pattern"""
        X, A, B, C, D = prices

        return {
            'XA': abs(A - X) / abs(C - X) if C != X else 0,
            'AB': abs(B - A) / abs(C - A) if C != A else 0,
            'BC': abs(C - B) / abs(A - B) if A != B else 0,
            'CD': abs(D - C) / abs(B - C) if B != C else 0
        }

    def _calculate_completion_price(self, prices: List[float], ratios: Dict) -> float:
        """Calculate expected completion price for the pattern"""
        X, A, B, C, D = prices

        # For bullish patterns, completion is above C
        # For bearish patterns, completion is below C
        if D > C:  # Bullish
            return C + (C - B) * ratios['CD']
        else:  # Bearish
            return C - (B - C) * ratios['CD']

    def _calculate_pattern_confidence(self, prices: List[float], target_ratios: Dict) -> float:
        """Calculate confidence score for the pattern"""
        actual_ratios = self._calculate_actual_ratios(prices)

        # Calculate how close actual ratios are to target ratios
        deviations = []
        for key in target_ratios.keys():
            if key in actual_ratios:
                deviation = abs(actual_ratios[key] - target_ratios[key])
                deviations.append(deviation)

        # Average deviation (lower is better)
        avg_deviation = np.mean(deviations)

        # Convert to confidence score (0-1, higher is better)
        confidence = max(0, 1 - avg_deviation * 2)  # Scale deviation to confidence

        return confidence


class ElliottWave:
    """Elliott Wave analysis and wave counting"""

    def __init__(self):
        self.wave_degrees = ['subminuette', 'minuette', 'minute', 'minor', 'intermediate']
        logger.info("Elliott Wave analyzer initialized")

    def analyze_waves(self, df: pd.DataFrame) -> Dict:
        """Analyze Elliott waves in the price data"""
        if len(df) < 100:
            return {'wave_count': None, 'current_wave': None, 'wave_degree': None}

        # Find swing points
        peaks, troughs = self._find_wave_points(df)

        # Attempt wave counting
        wave_structure = self._count_waves(df, peaks, troughs)

        # Determine current wave position
        current_position = self._determine_wave_position(wave_structure, df.iloc[-1]['close'])

        return {
            'wave_structure': wave_structure,
            'current_position': current_position,
            'wave_degree': self._determine_wave_degree(wave_structure),
            'confidence': self._calculate_wave_confidence(wave_structure)
        }

    def _find_wave_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Find significant wave points"""
        prices = df['close'].values

        # Use more sensitive peak detection for waves
        peaks, _ = find_peaks(prices, distance=5, prominence=np.std(prices) * 0.3)
        troughs, _ = find_peaks(-prices, distance=5, prominence=np.std(prices) * 0.3)

        return peaks.tolist(), troughs.tolist()

    def _count_waves(self, df: pd.DataFrame, peaks: List[int], troughs: List[int]) -> Dict:
        """Count Elliott waves"""
        # Combine and sort all significant points
        all_points = sorted(set(peaks + troughs))
        prices = [df.iloc[idx]['close'] for idx in all_points]

        if len(prices) < 5:
            return {}

        # Simple wave counting (this is a simplified version)
        wave_structure = {}

        # Identify impulse waves (1-2-3-4-5)
        impulse_waves = self._identify_impulse_waves(prices)

        # Identify corrective waves (A-B-C)
        corrective_waves = self._identify_corrective_waves(prices)

        wave_structure.update({
            'impulse_waves': impulse_waves,
            'corrective_waves': corrective_waves,
            'total_points': len(all_points)
        })

        return wave_structure

    def _identify_impulse_waves(self, prices: List[float]) -> List[Dict]:
        """Identify impulse wave patterns (1-2-3-4-5)"""
        impulse_patterns = []

        for i in range(len(prices) - 4):
            segment = prices[i:i+5]

            # Check for impulse wave characteristics
            if self._is_impulse_pattern(segment):
                pattern = {
                    'start_idx': i,
                    'waves': {
                        '1': segment[0],
                        '2': segment[1],
                        '3': segment[2],
                        '4': segment[3],
                        '5': segment[4]
                    },
                    'direction': 'up' if segment[-1] > segment[0] else 'down'
                }
                impulse_patterns.append(pattern)

        return impulse_patterns

    def _identify_corrective_waves(self, prices: List[float]) -> List[Dict]:
        """Identify corrective wave patterns (A-B-C)"""
        corrective_patterns = []

        for i in range(len(prices) - 2):
            segment = prices[i:i+3]

            # Check for zigzag correction
            if self._is_zigzag_correction(segment):
                pattern = {
                    'start_idx': i,
                    'waves': {
                        'A': segment[0],
                        'B': segment[1],
                        'C': segment[2]
                    },
                    'direction': 'up' if segment[-1] > segment[0] else 'down'
                }
                corrective_patterns.append(pattern)

        return corrective_patterns

    def _is_impulse_pattern(self, segment: List[float]) -> bool:
        """Check if segment follows impulse wave pattern"""
        if len(segment) != 5:
            return False

        w1, w2, w3, w4, w5 = segment

        # Wave 2 should not go beyond wave 1 start
        if w1 < w5:  # Uptrend
            wave2_rule = w2 < w1
            wave3_rule = w3 > w1 and w3 > w2
            wave4_rule = w4 > w2 and w4 < w3
            wave5_rule = w5 > w3
        else:  # Downtrend
            wave2_rule = w2 > w1
            wave3_rule = w3 < w1 and w3 < w2
            wave4_rule = w4 < w2 and w4 > w3
            wave5_rule = w5 < w3

        return all([wave2_rule, wave3_rule, wave4_rule, wave5_rule])

    def _is_zigzag_correction(self, segment: List[float]) -> bool:
        """Check if segment is a zigzag correction"""
        if len(segment) != 3:
            return False

        A, B, C = segment

        # B should not go beyond A start
        if A < C:  # Correcting downtrend
            return B < A and C > B
        else:  # Correcting uptrend
            return B > A and C < B

    def _determine_wave_position(self, wave_structure: Dict, current_price: float) -> str:
        """Determine current position in wave count"""
        if not wave_structure:
            return "unknown"

        # Simple position determination based on recent waves
        impulse_waves = wave_structure.get('impulse_waves', [])
        corrective_waves = wave_structure.get('corrective_waves', [])

        if impulse_waves:
            last_impulse = impulse_waves[-1]
            wave_5 = last_impulse['waves']['5']

            if abs(current_price - wave_5) / wave_5 < 0.01:  # Within 1% of wave 5
                return "wave_5_completion"
            elif current_price > wave_5:
                return "post_wave_5"
            else:
                return "wave_5"

        if corrective_waves:
            last_corrective = corrective_waves[-1]
            wave_c = last_corrective['waves']['C']

            if abs(current_price - wave_c) / wave_c < 0.01:
                return "wave_c_completion"
            else:
                return "wave_c"

        return "transition"

    def _determine_wave_degree(self, wave_structure: Dict) -> str:
        """Determine the degree of waves being analyzed"""
        total_points = wave_structure.get('total_points', 0)

        if total_points > 20:
            return 'intermediate'
        elif total_points > 15:
            return 'minor'
        elif total_points > 10:
            return 'minute'
        elif total_points > 5:
            return 'minuette'
        else:
            return 'subminuette'

    def _calculate_wave_confidence(self, wave_structure: Dict) -> float:
        """Calculate confidence in wave count"""
        if not wave_structure:
            return 0.0

        impulse_count = len(wave_structure.get('impulse_waves', []))
        corrective_count = len(wave_structure.get('corrective_waves', []))

        # Higher confidence with more consistent wave patterns
        total_patterns = impulse_count + corrective_count

        if total_patterns > 5:
            return 0.8
        elif total_patterns > 3:
            return 0.6
        elif total_patterns > 1:
            return 0.4
        else:
            return 0.2


class FibonacciAnalysis:
    """Fibonacci retracements, extensions, and projections"""

    def __init__(self):
        self.fib_levels = {
            'retracements': [0.236, 0.382, 0.5, 0.618, 0.786],
            'extensions': [1.0, 1.272, 1.414, 1.618, 2.0, 2.618],
            'projections': [0.618, 1.0, 1.618, 2.618]
        }
        logger.info("Fibonacci Analysis module initialized")

    def calculate_fibonacci_levels(self, df: pd.DataFrame) -> Dict:
        """Calculate Fibonacci levels for current trend"""
        if len(df) < 20:
            return {}

        # Find recent swing high and low
        recent_high = df['high'].tail(50).max()
        recent_low = df['low'].tail(50).min()
        current_price = df.iloc[-1]['close']

        # Determine trend direction
        trend = self._determine_trend(df)

        fib_levels = {}

        if trend == 'uptrend':
            # Calculate retracements from recent swing low to high
            fib_levels['retracements'] = self._calculate_retracements(recent_low, recent_high)
            # Calculate extensions from recent swing low to high
            fib_levels['extensions'] = self._calculate_extensions(recent_low, recent_high)
        else:
            # Calculate retracements from recent swing high to low
            fib_levels['retracements'] = self._calculate_retracements(recent_high, recent_low)
            # Calculate extensions from recent swing high to low
            fib_levels['extensions'] = self._calculate_extensions(recent_high, recent_low)

        # Find nearest Fibonacci level
        nearest_level = self._find_nearest_fib_level(current_price, fib_levels)

        return {
            'trend': trend,
            'swing_high': recent_high,
            'swing_low': recent_low,
            'fib_levels': fib_levels,
            'nearest_level': nearest_level,
            'current_price': current_price
        }

    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine current trend direction"""
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        current_price = df.iloc[-1]['close']

        if current_price > sma_20 > sma_50:
            return 'uptrend'
        elif current_price < sma_20 < sma_50:
            return 'downtrend'
        else:
            return 'sideways'

    def _calculate_retracements(self, low: float, high: float) -> Dict[float, float]:
        """Calculate Fibonacci retracement levels"""
        retracements = {}
        range_size = high - low

        for level in self.fib_levels['retracements']:
            if high > low:  # Uptrend
                retracements[level] = high - (range_size * level)
            else:  # Downtrend
                retracements[level] = low + (range_size * level)

        return retracements

    def _calculate_extensions(self, low: float, high: float) -> Dict[float, float]:
        """Calculate Fibonacci extension levels"""
        extensions = {}
        range_size = abs(high - low)

        for level in self.fib_levels['extensions']:
            if high > low:  # Uptrend
                extensions[level] = low + (range_size * level)
            else:  # Downtrend
                extensions[level] = high - (range_size * level)

        return extensions

    def _find_nearest_fib_level(self, price: float, fib_levels: Dict) -> Dict:
        """Find the nearest Fibonacci level to current price"""
        all_levels = []

        # Collect all fib levels
        for level_type, levels in fib_levels.items():
            if isinstance(levels, dict):
                for fib_ratio, level_price in levels.items():
                    all_levels.append({
                        'ratio': fib_ratio,
                        'price': level_price,
                        'type': level_type,
                        'distance': abs(price - level_price),
                        'distance_pct': abs(price - level_price) / price * 100
                    })

        if not all_levels:
            return {}

        # Find nearest level
        nearest = min(all_levels, key=lambda x: x['distance'])

        return nearest

    def get_fib_signals(self, fib_analysis: Dict) -> Dict:
        """Generate trading signals based on Fibonacci analysis"""
        if not fib_analysis:
            return {'signal': 'NEUTRAL', 'strength': 0}

        nearest_level = fib_analysis.get('nearest_level', {})
        current_price = fib_analysis.get('current_price', 0)
        trend = fib_analysis.get('trend', 'sideways')

        if not nearest_level:
            return {'signal': 'NEUTRAL', 'strength': 0}

        distance_pct = nearest_level.get('distance_pct', 100)
        level_type = nearest_level.get('type', '')
        fib_ratio = nearest_level.get('ratio', 0)

        signal = 'NEUTRAL'
        strength = 0

        # Generate signals based on proximity to fib levels
        if distance_pct < 0.5:  # Within 0.5% of fib level
            strength = 3  # Strong signal
        elif distance_pct < 1.0:  # Within 1% of fib level
            strength = 2  # Medium signal
        elif distance_pct < 2.0:  # Within 2% of fib level
            strength = 1  # Weak signal

        if strength > 0:
            if level_type == 'retracements':
                # At retracement level - potential reversal
                if trend == 'uptrend':
                    signal = 'BEARISH'  # Potential pullback in uptrend
                else:
                    signal = 'BULLISH'  # Potential bounce in downtrend
            elif level_type == 'extensions':
                # At extension level - potential continuation or reversal
                if fib_ratio >= 1.618:  # Strong extension
                    if trend == 'uptrend':
                        signal = 'BULLISH'  # Potential continuation
                    else:
                        signal = 'BEARISH'  # Potential continuation

        return {
            'signal': signal,
            'strength': strength,
            'nearest_level': nearest_level,
            'trend': trend
        }


class AdvancedPatternRecognition:
    """Main class for advanced pattern recognition combining all methods"""

    def __init__(self):
        self.harmonic = HarmonicPatterns()
        self.elliott = ElliottWave()
        self.fibonacci = FibonacciAnalysis()
        logger.info("Advanced Pattern Recognition initialized")

    def analyze_all_patterns(self, df: pd.DataFrame) -> Dict:
        """Perform complete pattern analysis"""
        if df is None or len(df) < 50:
            return {}

        analysis = {
            'harmonic_patterns': self.harmonic.detect_patterns(df),
            'elliott_waves': self.elliott.analyze_waves(df),
            'fibonacci_analysis': self.fibonacci.calculate_fibonacci_levels(df),
            'fibonacci_signals': {},
            'combined_signal': 'NEUTRAL',
            'confidence': 0.0
        }

        # Generate Fibonacci signals
        analysis['fibonacci_signals'] = self.fibonacci.get_fib_signals(
            analysis['fibonacci_analysis']
        )

        # Combine all signals into final recommendation
        analysis['combined_signal'] = self._combine_all_signals(analysis)
        analysis['confidence'] = self._calculate_overall_confidence(analysis)

        return analysis

    def _combine_all_signals(self, analysis: Dict) -> str:
        """Combine signals from all pattern recognition methods"""
        signals = []

        # Harmonic patterns signal
        harmonic_patterns = analysis.get('harmonic_patterns', {})
        for pattern_type, patterns in harmonic_patterns.items():
            if patterns:
                # Take the highest confidence pattern
                best_pattern = max(patterns, key=lambda x: x.get('confidence', 0))
                if best_pattern.get('confidence', 0) > 0.7:
                    signals.append(best_pattern.get('direction', 'neutral'))

        # Elliott wave signal
        elliott = analysis.get('elliott_waves', {})
        current_position = elliott.get('current_position', '')

        if 'completion' in current_position:
            if 'wave_5' in current_position:
                signals.append('bearish')  # Wave 5 completion often signals reversal
            elif 'wave_c' in current_position:
                signals.append('bullish')  # Wave C completion often signals reversal

        # Fibonacci signal
        fib_signal = analysis.get('fibonacci_signals', {}).get('signal', 'NEUTRAL')
        if fib_signal != 'NEUTRAL':
            signals.append(fib_signal.lower())

        # Count signals
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')

        if bullish_count > bearish_count:
            return 'BULLISH'
        elif bearish_count > bullish_count:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _calculate_overall_confidence(self, analysis: Dict) -> float:
        """Calculate overall confidence in pattern analysis"""
        confidence_scores = []

        # Harmonic patterns confidence
        harmonic_patterns = analysis.get('harmonic_patterns', {})
        for pattern_type, patterns in harmonic_patterns.items():
            if patterns:
                best_confidence = max(p.get('confidence', 0) for p in patterns)
                confidence_scores.append(best_confidence)

        # Elliott wave confidence
        elliott_confidence = analysis.get('elliott_waves', {}).get('confidence', 0)
        if elliott_confidence > 0:
            confidence_scores.append(elliott_confidence)

        # Fibonacci signal strength
        fib_strength = analysis.get('fibonacci_signals', {}).get('strength', 0)
        if fib_strength > 0:
            confidence_scores.append(fib_strength / 3.0)  # Normalize to 0-1

        if not confidence_scores:
            return 0.0

        # Average confidence
        return np.mean(confidence_scores)

    def get_pattern_signals(self, df: pd.DataFrame) -> Dict:
        """Get trading signals from pattern analysis"""
        analysis = self.analyze_all_patterns(df)

        return {
            'overall_signal': analysis.get('combined_signal', 'NEUTRAL'),
            'confidence': analysis.get('confidence', 0.0),
            'harmonic_patterns': analysis.get('harmonic_patterns', {}),
            'elliott_position': analysis.get('elliott_waves', {}).get('current_position'),
            'fibonacci_level': analysis.get('fibonacci_analysis', {}).get('nearest_level'),
            'analysis_details': analysis
        }


# Singleton instance
pattern_recognition = AdvancedPatternRecognition()
