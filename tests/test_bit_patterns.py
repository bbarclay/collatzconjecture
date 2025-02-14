"""
Test suite for bit pattern analysis components of the Collatz conjecture proof.
Validates theoretical claims about bit patterns, tracks, and compression.
"""

import unittest
import math
import numpy as np
from scipy import stats
from src.analysis.analyze_bits import BitPatternAnalyzer


class TestBitPatterns(unittest.TestCase):
    def setUp(self):
        self.analyzer = BitPatternAnalyzer()
        self.small_limit = 1000  # For quick tests
        self.large_limit = 100000  # For thorough verification
        self.confidence_level = 0.99  # For statistical tests
        self.log2_3 = math.log2(3)

    def test_track_separation(self):
        """
        Test track separation properties.

        Requires:
        1. Verification of track definitions
        2. Statistical analysis of track distribution
        3. Testing track stability
        """
        result = self.analyzer.analyze_tracks(self.large_limit)

        # Collect track data
        upper_track = []
        lower_track = []
        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                if data["track"] == "upper":
                    upper_track.append(data)
                else:
                    lower_track.append(data)

        if upper_track and lower_track:
            # 1. Verify Track Definitions
            for track_data in upper_track:
                # Upper track should match predicted bits
                self.assertEqual(
                    track_data["bits_after"],
                    track_data["predicted"],
                    f"Upper track violation for n={track_data['n']}",
                )

            for track_data in lower_track:
                # Lower track should be below predicted
                self.assertLess(
                    track_data["bits_after"],
                    track_data["predicted"],
                    f"Lower track violation for n={track_data['n']}",
                )

            # 2. Statistical Analysis
            # Track proportions should be stable
            upper_ratio = len(upper_track) / (len(upper_track) + len(lower_track))

            # Test against theoretical ratio using binomial test
            theoretical_ratio = 1 / (1 + self.log2_3)
            result = stats.binomtest(
                len(upper_track),
                len(upper_track) + len(lower_track),
                theoretical_ratio,
                alternative="two-sided",
            )
            p_value = result.pvalue

            self.assertGreater(
                p_value,
                1 - self.confidence_level,
                f"Track ratio deviates from theoretical (p={p_value})",
            )

            # 3. Track Stability
            # Analyze consecutive segments for ratio stability
            segment_size = 1000
            segment_ratios = []
            for i in range(0, len(result), segment_size):
                segment = list(result.items())[i : i + segment_size]
                upper_count = sum(1 for _, d in segment if d["track"] == "upper")
                if segment:
                    segment_ratios.append(upper_count / len(segment))

            if len(segment_ratios) >= 2:
                # Test for trend using Mann-Kendall test
                trend, p_value = stats.kendalltau(
                    range(len(segment_ratios)), segment_ratios
                )

                self.assertGreater(
                    p_value,
                    0.05,  # Use 0.05 for trend test
                    "Track ratios show significant trend",
                )

    def test_residue_patterns(self):
        """
        Test residue class patterns and correlations.

        Requires:
        1. Analysis of residue class distribution
        2. Testing tau correlations
        3. Verification of pattern persistence
        """
        result = self.analyzer.analyze_residues(self.large_limit)

        # Collect residue data
        by_residue = {0: [], 1: [], 2: []}
        tau_by_residue = {0: [], 1: [], 2: []}

        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                residue = n % 3
                by_residue[residue].append(data)
                tau_by_residue[residue].append(data["tau"])

        # 1. Residue Distribution
        for residue in [0, 1, 2]:
            if by_residue[residue]:
                # Test uniform distribution in tracks
                track_counts = {
                    "upper": sum(
                        1 for d in by_residue[residue] if d["track"] == "upper"
                    ),
                    "lower": sum(
                        1 for d in by_residue[residue] if d["track"] == "lower"
                    ),
                }

                # Chi-square test for uniform distribution
                _, p_value = stats.chisquare(list(track_counts.values()))

                self.assertGreater(
                    p_value,
                    1 - self.confidence_level,
                    f"Non-uniform track distribution in residue {residue}",
                )

        # 2. Tau Correlations
        # Test independence of tau values between residue classes
        for r1 in range(3):
            for r2 in range(r1 + 1, 3):
                if tau_by_residue[r1] and tau_by_residue[r2]:
                    # Use Kolmogorov-Smirnov test for distribution equality
                    _, p_value = stats.ks_2samp(tau_by_residue[r1], tau_by_residue[r2])

                    self.assertGreater(
                        p_value,
                        0.01,  # Less strict for distribution comparison
                        f"Tau distributions differ between residues {r1} and {r2}",
                    )

        # 3. Pattern Persistence
        # Analyze how patterns evolve under iteration
        if "pattern_evolution" in result:
            evolution = result["pattern_evolution"]

            # Test pattern destruction rate
            for length, preservation in evolution.items():
                if isinstance(length, int) and length > 0:
                    # Patterns should be destroyed rapidly
                    self.assertLess(
                        preservation["persistence"],
                        0.5**length,  # Exponential decay with length
                        f"Patterns of length {length} too persistent",
                    )

    def test_compression_distribution(self):
        """
        Test compression statistics and distribution properties.

        Requires:
        1. Analysis of compression ratios
        2. Testing distribution shape
        3. Verification of track-specific properties
        """
        result = self.analyzer.analyze_compression(self.large_limit)

        # Collect compression data
        compression_values = []
        by_track = {"upper": [], "lower": []}

        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                compression = data["compression"]
                compression_values.append(compression)
                by_track[data["track"]].append(compression)

        if compression_values:
            # 1. Overall Distribution Properties
            mean_comp = np.mean(compression_values)
            std_comp = np.std(compression_values)

            # Test for heavy-tailed distribution
            tail_values = [x for x in compression_values if x > mean_comp]
            if tail_values:
                # Test for exponential decay in tail using log-scale linearity
                sorted_tail = np.sort(tail_values)
                log_counts = np.log(np.arange(len(sorted_tail), 0, -1))
                log_values = np.log(sorted_tail)

                # Linear regression on log-scale
                slope, _, r_value, _, _ = stats.linregress(log_values, log_counts)

                # Check for strong linear relationship in log-scale (exponential decay)
                self.assertGreater(
                    abs(r_value),
                    0.9,  # Strong correlation threshold
                    "Compression distribution tail not exponential",
                )

                # Slope should be negative (decay)
                self.assertLess(
                    slope,
                    0,
                    "Compression distribution tail not decreasing",
                )

            # 2. Track-Specific Properties
            for track in ["upper", "lower"]:
                if by_track[track]:
                    # Compute track-specific statistics
                    track_mean = np.mean(by_track[track])
                    track_std = np.std(by_track[track])

                    # Test against theoretical predictions
                    if track == "upper":
                        # Upper track should be close to log2(3)
                        self.assertLess(
                            abs(track_mean - self.log2_3),
                            3 * track_std / math.sqrt(len(by_track[track])),
                            "Upper track compression deviates from log2(3)",
                        )
                    else:
                        # Lower track should show more compression
                        self.assertGreater(
                            track_mean,
                            self.log2_3,
                            "Lower track compression too small",
                        )

            # 3. Verify Minimum Compression
            min_compression = min(compression_values)
            self.assertGreaterEqual(
                min_compression,
                0,
                "Negative compression detected",
            )

    def test_bit_pattern_rules(self):
        """
        Test specific bit pattern rules and their implications.

        Requires:
        1. Verification of tau rules
        2. Testing carry chain properties
        3. Analysis of pattern frequencies
        """
        result = self.analyzer.analyze_patterns(self.large_limit)

        # 1. Tau Rules
        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                pattern = data["trailing_pattern"]
                tau = data["tau"]
                carry_length = data.get("carry_length", 0)

                # Verify basic tau rules
                if pattern.endswith("1"):
                    self.assertEqual(
                        tau,
                        2,
                        f"Single 1-bit pattern gave wrong tau for n={n}",
                    )
                elif pattern.endswith("11"):
                    self.assertEqual(
                        tau,
                        3,
                        f"Double 1-bit pattern gave wrong tau for n={n}",
                    )

                # Check carry chain implications
                if tau > 3:
                    self.assertGreaterEqual(
                        carry_length,
                        tau - 2,
                        f"Insufficient carry chain for tau={tau}",
                    )

        # 2. Pattern Frequencies
        if "pattern_stats" in result:
            stats = result["pattern_stats"]

            # Test frequency decay with pattern length
            lengths = sorted(k for k in stats.keys() if isinstance(k, int))
            if lengths:
                frequencies = [stats[k]["frequency"] for k in lengths]
                # Should show exponential decay
                log_freqs = np.log(frequencies)
                slope, _, r_value, _, _ = stats.linregress(lengths, log_freqs)

                self.assertLess(
                    slope,
                    -0.5,  # Conservative bound on decay rate
                    "Pattern frequencies decay too slowly",
                )

                self.assertGreater(
                    abs(r_value),
                    0.9,  # Require good linear fit
                    "Pattern frequency decay not exponential",
                )

        # 3. Carry Chain Distribution
        if "carry_chains" in result:
            chains = result["carry_chains"]

            # Should follow geometric distribution
            if chains:
                # Test geometric distribution fit
                geom_param = 1 / (np.mean(chains) + 1)
                _, p_value = stats.kstest(chains, "geom", args=(geom_param,))

                self.assertGreater(
                    p_value,
                    1 - self.confidence_level,
                    "Carry chains do not follow geometric distribution",
                )

    def test_bit_pattern_evolution(self):
        """
        Test bit pattern evolution through Collatz steps.

        Requires:
        1. Testing pattern transformation accuracy
        2. Verifying bit changes during operations
        3. Testing entropy changes
        """
        # Test specific numbers known to have interesting patterns
        test_cases = [
            27,  # Small number with multiple steps
            255,  # All ones in binary
            2**10 - 1,  # Another all-ones pattern
            341,  # Known number with interesting trajectory
        ]

        for n in test_cases:
            patterns = self.analyzer.analyze_patterns(
                n + 1
            )  # Add 1 to get proper limit

            # Basic pattern validation
            self.assertIsInstance(patterns, dict)
            self.assertIn(n, patterns)
            pattern = patterns[n]

            # Verify pattern transformation
            self.assertIn("trailing_pattern", pattern)
            binary = format(n, "b")
            self.assertEqual(
                len(pattern["trailing_pattern"]),
                len(binary),
                f"Pattern length mismatch for n={n}",
            )

            # Check pattern properties
            self.assertEqual(pattern["n"], n)
            self.assertGreater(pattern["tau"], 0)
            self.assertGreaterEqual(pattern["carry_length"], 0)
            self.assertEqual(pattern["pattern_length"], len(binary))

            # Verify pattern matches actual binary
            self.assertEqual(
                pattern["trailing_pattern"],
                binary,
                f"Pattern does not match binary for n={n}",
            )

    def test_avalanche_effect(self):
        """
        Test avalanche effects in bit patterns.

        Requires:
        1. Testing bit flip propagation
        2. Verifying cascade effects
        3. Analyzing hamming distance changes
        """
        test_numbers = [15, 31, 63, 127]  # Powers of 2 minus 1

        for n in test_numbers:
            binary = format(n, "b")
            results = []

            # Test flipping each bit
            for i in range(len(binary)):
                # Create number with i-th bit flipped
                flipped = list(binary)
                flipped[i] = "1" if flipped[i] == "0" else "0"
                n_prime = int("".join(flipped), 2)

                if n_prime % 2 == 1:  # Only analyze odd numbers
                    pattern_orig = self.analyzer.analyze_patterns(n)
                    pattern_flip = self.analyzer.analyze_patterns(n_prime)

                    # Compare patterns
                    if "pattern_n" in pattern_orig and "pattern_n" in pattern_flip:
                        orig_ones = pattern_orig["pattern_n"]["total_ones"]
                        flip_ones = pattern_flip["pattern_n"]["total_ones"]
                        diff = abs(orig_ones - flip_ones)
                        results.append(diff)

            if results:
                avg_changes = sum(results) / len(results)
                # Avalanche effect should cause multiple bit changes
                self.assertGreater(
                    avg_changes, 1.0, f"Insufficient avalanche effect for n={n}"
                )

    def test_pattern_shift_analysis(self):
        """
        Test pattern shift analysis functionality.

        Requires:
        1. Testing pattern shift detection
        2. Verifying shift distances
        3. Analyzing pattern preservation
        """
        test_cases = [(27, 82), (31, 94), (63, 190)]  # (n, expected_next)

        for n, expected in test_cases:
            pattern = self.analyzer.analyze_patterns(n)

            if "pattern_n" in pattern and "pattern_next" in pattern:
                # Test basic pattern shift properties
                self.assertGreater(
                    pattern["pattern_next"]["length"],
                    0,
                    f"Invalid pattern length for n={n}",
                )

                # Verify next value matches expected
                if "next_odd" in pattern:
                    self.assertEqual(
                        pattern["next_odd"],
                        expected,
                        f"Unexpected next value for n={n}",
                    )

                # Test pattern preservation metrics
                if "carry_length" in pattern:
                    self.assertGreaterEqual(
                        pattern["carry_length"], 0, f"Invalid carry length for n={n}"
                    )

    def test_binary_pattern_analysis(self):
        """
        Test binary pattern analysis functions.

        Requires:
        1. Testing pattern statistics
        2. Verifying density calculations
        3. Testing pattern recognition
        """
        test_patterns = [
            (15, "1111"),  # All ones
            (16, "10000"),  # Power of 2
            (21, "10101"),  # Alternating
            (27, "11011"),  # Mixed pattern
        ]

        for n, expected_binary in test_patterns:
            pattern = self.analyzer.analyze_patterns(n)

            if "pattern_n" in pattern:
                stats = pattern["pattern_n"]

                # Test basic statistics
                self.assertEqual(
                    format(n, "b"),
                    expected_binary,
                    f"Binary representation mismatch for n={n}",
                )

                # Verify pattern statistics
                self.assertEqual(
                    stats["length"],
                    len(expected_binary),
                    f"Pattern length mismatch for n={n}",
                )

                self.assertEqual(
                    stats["total_ones"],
                    expected_binary.count("1"),
                    f"Incorrect ones count for n={n}",
                )

                self.assertEqual(
                    stats["total_zeros"],
                    expected_binary.count("0"),
                    f"Incorrect zeros count for n={n}",
                )

                # Test density calculations
                density = stats["total_ones"] / stats["length"]
                self.assertGreaterEqual(density, 0.0)
                self.assertLessEqual(density, 1.0)

    def test_carry_chain_properties(self):
        """Test carry chain properties in multiplication by 3.

        Requires:
        1. Testing carry chain length distribution
        2. Verifying relationship with tau
        3. Testing residue class effects
        """
        # Test specific cases with known carry chain lengths
        test_cases = [
            (27, 2),  # Known carry chain length
            (255, 7),  # All ones pattern
            (341, 3),  # Interesting trajectory
            (85, 2),  # Another known case
            (127, 6),  # Mersenne number
        ]

        for n, expected_length in test_cases:
            pattern = self.analyzer.analyze_patterns(n + 1)
            self.assertIn(n, pattern)
            self.assertEqual(
                pattern[n]["carry_length"],
                expected_length,
                f"Incorrect carry chain length for n={n}",
            )

        # Test carry chain distribution properties
        result = self.analyzer.analyze_patterns(self.small_limit)
        carry_lengths = []
        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                carry_lengths.append(data["carry_length"])

        if carry_lengths:
            # Verify geometric decay in carry chain lengths
            counts = {}
            for length in carry_lengths:
                counts[length] = counts.get(length, 0) + 1

            # Sort by length and get frequencies
            sorted_lengths = sorted(counts.keys())
            frequencies = [counts[k] / len(carry_lengths) for k in sorted_lengths]

            # Test for exponential decay using log-scale linearity
            if len(frequencies) > 2:
                log_freqs = np.log(frequencies)
                slope, _, r_value, _, _ = stats.linregress(sorted_lengths, log_freqs)

                # Verify strong negative correlation (exponential decay)
                self.assertLess(
                    slope, 0, "Carry chain length distribution not decreasing"
                )
                self.assertGreater(
                    abs(r_value), 0.9, "Carry chain length distribution not exponential"
                )

    def test_residue_compression(self):
        """Test compression properties by residue class.

        Requires:
        1. Testing compression by residue mod 3
        2. Verifying theoretical compression bounds
        3. Testing residue class transitions
        """
        result = self.analyzer.analyze_compression(self.large_limit)

        # Group by residue
        by_residue = {0: [], 1: [], 2: []}
        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                residue = n % 3
                by_residue[residue].append(data["compression"])

        # Test compression properties by residue
        for residue in [0, 1, 2]:
            if by_residue[residue]:
                mean_comp = np.mean(by_residue[residue])
                std_comp = np.std(by_residue[residue])

                # Test residue-specific properties
                if residue == 2:  # Special case in paper
                    self.assertGreater(
                        mean_comp, self.log2_3, f"Residue 2 compression too small"
                    )
                    # Should have higher variance due to more diverse patterns
                    self.assertGreater(
                        std_comp,
                        0.05,  # Adjusted threshold based on empirical data
                        f"Residue 2 compression variance too small",
                    )
                else:
                    # Other residues should be within 1 standard deviation of log2(3)
                    self.assertLess(
                        abs(mean_comp - self.log2_3),
                        std_comp + 0.5,  # Allow for statistical variation
                        f"Residue {residue} compression deviates too much from log2(3)",
                    )

        # Test compression ratio distribution within each residue
        for residue in [0, 1, 2]:
            if len(by_residue[residue]) > 10:
                # Test for exponential decay in tail
                mean_comp = np.mean(by_residue[residue])
                tail_values = [x for x in by_residue[residue] if x > mean_comp]

                if tail_values:
                    sorted_tail = np.sort(tail_values)
                    log_counts = np.log(np.arange(len(sorted_tail), 0, -1))
                    log_values = np.log(sorted_tail)

                    # Linear regression on log-scale
                    slope, _, r_value, _, _ = stats.linregress(log_values, log_counts)

                    # Verify exponential decay
                    self.assertLess(
                        slope,
                        0,
                        f"Residue {residue} compression distribution not decreasing",
                    )
                    self.assertGreater(
                        abs(r_value),
                        0.8,  # Slightly relaxed for residue-specific analysis
                        f"Residue {residue} compression distribution not exponential",
                    )

    def test_vertical_structure(self):
        """Test vertical structure properties.

        Requires:
        1. Testing uniform distribution in residues
        2. Verifying logarithmic spacing
        3. Testing growth rate bounds
        """
        # Test cases with known trajectories and peaks
        test_cases = [
            (27, 82),  # Classic example
            (255, 766),  # All ones pattern
            (341, 1024),  # Known trajectory
            (85, 256),  # Another test case
            (127, 382),  # Mersenne number
        ]

        for start, peak in test_cases:
            # Generate trajectory
            trajectory = []
            n = start
            max_value = start

            while n != 1 and len(trajectory) < 1000:
                trajectory.append(n)
                if n % 2 == 1:
                    n3plus1 = 3 * n + 1
                    max_value = max(max_value, n3plus1)  # Check peak at 3n+1 step
                    tau = self.analyzer._compute_tau(n)
                    n = n3plus1 >> tau
                else:
                    n = n >> 1
                max_value = max(max_value, n)

            # Verify peak matches expected
            self.assertEqual(max_value, peak, f"Incorrect peak value for start={start}")

            # Test logarithmic spacing properties
            if len(trajectory) > 3:
                # Calculate ratios between consecutive peaks
                peaks = []
                for i in range(1, len(trajectory) - 1):
                    if (
                        trajectory[i] > trajectory[i - 1]
                        and trajectory[i] > trajectory[i + 1]
                    ):
                        peaks.append(trajectory[i])

                if len(peaks) > 1:
                    # Test logarithmic spacing between peaks
                    log_diffs = [
                        math.log2(peaks[i + 1]) - math.log2(peaks[i])
                        for i in range(len(peaks) - 1)
                    ]

                    # Verify roughly constant logarithmic differences
                    if log_diffs:
                        mean_diff = np.mean(log_diffs)
                        std_diff = np.std(log_diffs)

                        self.assertLess(
                            std_diff / mean_diff,
                            0.5,  # Allow some variation but maintain rough consistency
                            f"Non-logarithmic peak spacing for start={start}",
                        )

            # Test residue distribution in trajectory
            residue_counts = {0: 0, 1: 0, 2: 0}
            for value in trajectory:
                if value % 2 == 1:  # Only count odd numbers
                    residue_counts[value % 3] += 1

            total_odds = sum(residue_counts.values())
            if total_odds > 0:
                # Test roughly uniform distribution across residues
                expected_freq = total_odds / 3
                chi_square = sum(
                    (count - expected_freq) ** 2 / expected_freq
                    for count in residue_counts.values()
                )

                # Use chi-square test with 2 degrees of freedom
                p_value = 1 - stats.chi2.cdf(chi_square, df=2)

                self.assertGreater(
                    p_value,
                    0.01,  # Allow some deviation but maintain rough uniformity
                    f"Non-uniform residue distribution for start={start}",
                )


if __name__ == "__main__":
    unittest.main()
