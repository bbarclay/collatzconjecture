"""
Test suite for information theory verification components of the Collatz conjecture proof.
Validates theoretical claims about entropy, compression, and bit patterns.
"""

import unittest
import math
import numpy as np
from scipy import stats
from src.core.information_theory import InformationTheoryVerifier


class TestInformationTheory(unittest.TestCase):
    def setUp(self):
        self.verifier = InformationTheoryVerifier()
        self.small_limit = 1000  # For quick tests
        self.large_limit = 100000  # For thorough verification
        self.confidence_level = 0.99  # For statistical tests
        self.log2_3 = math.log2(3)

    def test_entropy_reduction_theorem(self):
        """
        Test Theorem entropy: Expected entropy change after one iteration is negative.

        This is a fundamental theorem that requires rigorous statistical validation:
        1. Check actual entropy changes against theoretical bounds
        2. Verify statistical significance of the reduction
        3. Test across different residue classes
        """
        result = self.verifier.verify_entropy_change(self.large_limit)

        # Collect entropy changes across different starting values
        entropy_changes = []
        theoretical_bounds = []

        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:  # Only odd numbers
                delta_h = data["delta_h"]
                theoretical = data["theoretical"]
                entropy_changes.append(delta_h)
                theoretical_bounds.append(theoretical)

        if entropy_changes:
            # 1. Statistical Test: Is mean entropy change negative?
            mean_change = np.mean(entropy_changes)
            std_change = np.std(entropy_changes, ddof=1)
            n_samples = len(entropy_changes)

            # Compute t-statistic and p-value for H0: mean ≥ 0
            t_stat = mean_change / (std_change / math.sqrt(n_samples))
            p_value = stats.t.cdf(t_stat, df=n_samples - 1)

            self.assertLess(
                p_value,
                1 - self.confidence_level,
                f"Entropy reduction not statistically significant (p={p_value})",
            )

            # 2. Verify Global Bounds
            for delta_h, bound in zip(entropy_changes, theoretical_bounds):
                # Upper bound: log_2(3) - τ(n)
                self.assertLessEqual(
                    delta_h,
                    self.log2_3,
                    "Entropy change exceeds theoretical upper bound",
                )

                # Check against provided theoretical bound
                self.assertLessEqual(
                    abs(delta_h - bound),
                    1e-10,  # Numerical precision tolerance
                    "Entropy change deviates from theoretical calculation",
                )

            # 3. Check Residue Classes
            by_residue = {0: [], 1: [], 2: []}
            for n, data in result.items():
                if isinstance(n, int) and n % 2 == 1:
                    by_residue[n % 3].append(data["delta_h"])

            # All residue classes should show negative mean change
            for residue, changes in by_residue.items():
                if changes:
                    mean_residue = np.mean(changes)
                    self.assertLess(
                        mean_residue,
                        0,
                        f"Residue class {residue} mod 3 shows positive entropy change",
                    )

    def test_compression_ratio_theorem(self):
        """
        Test Theorem compression_ratio: Average compression ratio is less than 1.

        Requires:
        1. Statistical verification of ratio < 1
        2. Proper confidence intervals
        3. Testing theoretical formula
        """
        result = self.verifier.verify_compression_ratio(self.large_limit)

        ratios = []
        tau_values = []

        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                ratios.append(data["ratio"])
                tau_values.append(data["tau"])

        if ratios:
            # 1. Statistical Test: Is mean ratio < 1?
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios, ddof=1)
            n_samples = len(ratios)

            # Compute confidence interval
            ci_upper = mean_ratio + stats.t.ppf(
                self.confidence_level, df=n_samples - 1
            ) * (std_ratio / math.sqrt(n_samples))

            self.assertLess(
                ci_upper,
                1.0,
                f"Compression ratio not significantly less than 1 (CI upper={ci_upper})",
            )

            # 2. Verify Theoretical Formula
            mean_tau = np.mean(tau_values)
            theoretical_ratio = self.log2_3 / mean_tau

            # Compare empirical and theoretical ratios
            self.assertLess(
                abs(mean_ratio - theoretical_ratio),
                3 * std_ratio / math.sqrt(n_samples),  # 3σ confidence
                "Empirical ratio deviates significantly from theoretical prediction",
            )

    def test_global_descent_theorem(self):
        """
        Test Theorem global_descent: Trajectories must eventually descend.

        Requires:
        1. Statistical verification of descent probability
        2. Analysis of descent times
        3. Verification across different starting ranges
        """
        result = self.verifier.verify_global_descent(self.large_limit)

        descent_times = []
        max_excursions = []

        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                if data["descends"]:
                    descent_times.append(data["steps_to_descent"])
                max_excursions.append(data["max_excursion"])

        if descent_times:
            # 1. Verify High Descent Probability
            descent_prob = len(descent_times) / len(max_excursions)

            # Binomial test for H0: p ≤ 0.99
            p_value = 1 - stats.binom.cdf(
                len(descent_times) - 1, len(max_excursions), 0.99
            )

            self.assertLess(
                p_value,
                1 - self.confidence_level,
                f"Descent probability not significant (p={p_value})",
            )

            # 2. Analyze Descent Times
            # Should follow exponential distribution
            _, p_value = stats.kstest(
                descent_times, "expon", args=(0, np.mean(descent_times))
            )

            self.assertGreater(
                p_value,
                1 - self.confidence_level,
                "Descent times do not follow exponential distribution",
            )

            # 3. Check Max Excursions
            # Should be bounded by theoretical prediction
            max_exc = max(max_excursions)
            theoretical_bound = math.log2(self.large_limit) * 2  # Conservative bound

            self.assertLess(
                max_exc,
                theoretical_bound,
                "Maximum excursion exceeds theoretical bound",
            )

    def test_bit_pattern_analysis(self):
        """
        Test bit pattern properties and τ determination rules.

        Requires:
        1. Verification of τ rules
        2. Analysis of carry chain effects
        3. Pattern destruction verification
        """
        result = self.verifier.analyze_bit_patterns(self.large_limit)

        # 1. Verify τ Rules
        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                pattern = data["trailing_pattern"]
                tau = data["tau"]
                carry_length = data["carry_length"]

                # Single 1-bit should give τ = 2
                if pattern == "1":
                    self.assertEqual(
                        tau,
                        2,
                        f"Single 1-bit pattern gave τ={tau}, expected 2",
                    )

                # Two 1-bits should give τ = 3
                elif pattern == "11":
                    self.assertEqual(
                        tau,
                        3,
                        f"Double 1-bit pattern gave τ={tau}, expected 3",
                    )

                # Long carry track implies carry chain ≥ 2
                if tau > 3:
                    self.assertGreaterEqual(
                        carry_length,
                        2,
                        f"High τ={tau} without long carry chain",
                    )

        # 2. Verify Pattern Destruction
        pattern_preservation = result.get("pattern_preservation", {})
        for length, preserved_ratio in pattern_preservation.items():
            if isinstance(length, int) and length > 0:
                # Multiplication by 3 should destroy most patterns
                self.assertLess(
                    preserved_ratio,
                    0.1,  # Allow some coincidental preservation
                    f"Too many {length}-bit patterns preserved",
                )

        # 3. Check Carry Chain Distribution
        carry_lengths = [
            data["carry_length"]
            for data in result.values()
            if isinstance(data, dict) and "carry_length" in data
        ]

        if carry_lengths:
            # Should follow geometric distribution
            geom_param = 1 / (np.mean(carry_lengths) + 1)
            _, p_value = stats.kstest(carry_lengths, "geom", args=(geom_param,))

            self.assertGreater(
                p_value,
                1 - self.confidence_level,
                "Carry chain lengths do not follow geometric distribution",
            )

    def test_global_entropy_bounds(self):
        """
        Test global entropy bounds and exact inequalities.

        Requires:
        1. Verification of upper and lower bounds
        2. Testing bound tightness
        3. Checking error term behavior
        """
        result = self.verifier.verify_entropy_bounds(self.large_limit)

        for n, data in result.items():
            if isinstance(n, int) and n % 2 == 1:
                delta_h = data["delta_h"]
                tau = data["tau"]

                # 1. Verify Bounds
                upper_bound = self.log2_3 - tau
                lower_bound = upper_bound - 1 / (3 * n * math.log(2))

                self.assertLessEqual(
                    delta_h,
                    upper_bound,
                    f"Upper bound violated for n={n}",
                )

                self.assertGreaterEqual(
                    delta_h,
                    lower_bound,
                    f"Lower bound violated for n={n}",
                )

                # 2. Check Error Term
                error = upper_bound - delta_h
                theoretical_error = 1 / (3 * n * math.log(2))

                self.assertLessEqual(
                    error,
                    theoretical_error,
                    f"Error term exceeds theoretical bound for n={n}",
                )

        # 3. Verify Asymptotic Behavior
        if "asymptotic" in result:
            errors = result["asymptotic"]["errors"]
            ns = result["asymptotic"]["ns"]

            # Fit power law to errors
            log_errors = np.log(errors)
            log_ns = np.log(ns)
            slope, _, r_value, _, _ = stats.linregress(log_ns, log_errors)

            # Should decay at least as O(1/n)
            self.assertLessEqual(
                slope,
                -0.9,  # Allow some numerical error from -1
                "Error terms do not show O(1/n) decay",
            )

            # Check fit quality
            self.assertGreaterEqual(
                abs(r_value),
                0.95,
                "Poor power law fit for error term decay",
            )


if __name__ == "__main__":
    unittest.main()
