"""
Test suite for Collatz conjecture verification code.
"""

import unittest
import math
from src.core.collatz_verifier import CollatzVerifier
from src.core.bit_pattern_analyzer import BitPatternAnalyzer
from src.core.information_theory import InformationTheoryVerifier
from src.core.measure_theory import MeasureTheoryVerifier
from src.core.theorem_mapper import TheoremMapper


class TestCollatzVerifier(unittest.TestCase):
    def setUp(self):
        self.verifier = CollatzVerifier()

    def test_collatz_step(self):
        """Test basic Collatz step function"""
        self.assertEqual(self.verifier.collatz_step(1), 4)
        self.assertEqual(self.verifier.collatz_step(4), 2)
        self.assertEqual(self.verifier.collatz_step(2), 1)
        self.assertEqual(self.verifier.collatz_step(5), 16)

    def test_trajectory(self):
        """Test trajectory computation"""
        path, max_reached = self.verifier.trajectory(5)
        self.assertEqual(path, [5, 16, 8, 4, 2, 1])
        self.assertFalse(max_reached)

    def test_find_tau(self):
        """Test τ computation"""
        self.assertEqual(self.verifier.find_tau(1), 2)  # 3×1 + 1 = 4 = 2^2
        self.assertEqual(self.verifier.find_tau(3), 1)  # 3×3 + 1 = 10 = 2×5
        with self.assertRaises(ValueError):
            self.verifier.find_tau(2)  # Even numbers not allowed

    def test_verify_no_even_cycles(self):
        """Test verification of no even cycles"""
        result, cycle = self.verifier.verify_no_even_cycles(100)
        self.assertTrue(result)
        self.assertIsNone(cycle)

    def test_verify_forward_uniqueness(self):
        """Test forward uniqueness property"""
        result, collision = self.verifier.verify_forward_uniqueness(100)
        self.assertTrue(result)
        self.assertIsNone(collision)


class TestBitPatternAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = BitPatternAnalyzer()

    def test_analyze_bit_patterns(self):
        """Test bit pattern analysis"""
        upper, lower = self.analyzer.analyze_bit_patterns(range(1, 10, 2))
        self.assertTrue(len(upper) + len(lower) > 0)

    def test_track_spacing(self):
        """Test track spacing computation"""
        spacing = self.analyzer.compute_track_spacing((3, 5), (3, 4))
        self.assertEqual(spacing, 1)


class TestInformationTheoryVerifier(unittest.TestCase):
    def setUp(self):
        self.info_verifier = InformationTheoryVerifier()

    def test_entropy_change(self):
        """Test entropy change computation"""
        result = self.info_verifier.entropy_change(3)
        self.assertIn("delta_h", result)
        self.assertIn("theoretical", result)
        self.assertIn("error", result)

    def test_entropy_bounds(self):
        """Test entropy bound verification"""
        result = self.info_verifier.verify_entropy_bounds(3)
        self.assertIn("actual", result)
        self.assertIn("upper_bound", result)
        self.assertIn("lower_bound", result)
        self.assertIn("within_bounds", result)


class TestMeasureTheoryVerifier(unittest.TestCase):
    def setUp(self):
        self.measure_verifier = MeasureTheoryVerifier()

    def test_tau_distribution(self):
        """Test τ distribution verification including residue patterns"""
        result = self.measure_verifier.verify_tau_distribution(100)

        # Check basic structure
        self.assertIn("basic", result)
        self.assertIn("residue_3", result)
        self.assertIn("global", result)

        # Check theoretical bounds
        self.assertTrue(
            result["global"]["theoretical_matches"], "Error terms should be O(n^(-1/2))"
        )

        # Check residue patterns
        for tau, pattern in result["residue_3"].items():
            self.assertLess(
                pattern["variation"], 0.5, "Residue patterns should be roughly uniform"
            )

    def test_measure_preservation(self):
        """Test measure preservation on arithmetic progressions"""
        result = self.measure_verifier.verify_measure_preservation(100)

        # Print debug information
        print("\nMeasure preservation test results:")
        for prog_key, prog_result in result.items():
            if prog_key != "global":
                print(f"\n{prog_key}:")
                print(f"  set_measure: {prog_result['set_measure']:.4f}")
                print(f"  preimage_measure: {prog_result['preimage_measure']:.4f}")
                print(f"  difference: {prog_result['difference']:.4f}")
                print(f"  is_preserved: {prog_result['is_preserved']}")

                if "tau_analysis" in prog_result:
                    print("  Tau analysis:")
                    for k, tau_result in prog_result["tau_analysis"].items():
                        print(f"    tau={k}:")
                        print(f"      measure: {tau_result['measure']:.4f}")
                        print(f"      expected: {tau_result['expected']:.4f}")
                        print(f"      difference: {tau_result['difference']:.4f}")
                        print(f"      is_preserved: {tau_result['is_preserved']}")

        # Check preservation for arithmetic progressions
        progression_results = {k: v for k, v in result.items() if k != "global"}

        self.assertTrue(
            any(r["is_preserved"] for r in progression_results.values()),
            "At least one arithmetic progression should preserve measure",
        )

        # Check tau-specific preservation
        for prog_result in progression_results.values():
            if "tau_analysis" in prog_result:
                tau_results = prog_result["tau_analysis"]
                self.assertTrue(
                    any(t["is_preserved"] for t in tau_results.values() if t),
                    "At least one tau value should preserve measure",
                )

        # Check global convergence
        self.assertIn("global", result)
        self.assertLess(
            result["global"]["max_difference"],
            0.5,
            "Should have bounded difference across progressions",
        )

    def test_ergodicity(self):
        """Test strong mixing properties"""
        result = self.measure_verifier.verify_ergodicity(100)

        # Print debug information
        print("\nErgodicity test results:")
        for prog_key, prog_result in result.items():
            if prog_key != "global":
                print(f"\n{prog_key}:")
                print(f"  decay_rate: {prog_result['decay_rate']:.4f}")
                print(f"  is_exponential: {prog_result['is_exponential']}")
                print(f"  sample_size: {prog_result['sample_size']}")
                print("  Correlations:")
                for lag, corr in prog_result["correlations"].items():
                    print(f"    lag {lag}: {corr:.4f}")

        if "global" in result:
            print("\nGlobal statistics:")
            print(f"  avg_decay_rate: {result['global']['avg_decay_rate']:.4f}")
            print(f"  min_decay_rate: {result['global']['min_decay_rate']:.4f}")
            print(f"  max_decay_rate: {result['global']['max_decay_rate']:.4f}")
            print(f"  strong_mixing: {result['global']['strong_mixing']}")

        # Check correlation decay
        self.assertIn("global", result)
        self.assertTrue(
            result["global"]["strong_mixing"], "Should exhibit strong mixing"
        )

        # Check decay rates
        self.assertGreater(
            result["global"]["min_decay_rate"], 0, "Should have positive decay rate"
        )

        # Check individual progressions
        progression_results = {k: v for k, v in result.items() if k != "global"}
        for prog_result in progression_results.values():
            self.assertIn("correlations", prog_result)
            self.assertIn("decay_rate", prog_result)
            self.assertIn("is_exponential", prog_result)


class TestTheoremMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = TheoremMapper()

    def test_theorem_verification(self):
        """Test theorem verification"""
        result = self.mapper.verify_theorem("thm:one_way")
        self.assertIn("forward_complexity", result)
        self.assertIn("predecessor_space", result)

    def test_dependencies(self):
        """Test theorem dependency resolution"""
        result = self.mapper.verify_theorem("thm:global_descent")
        self.assertIn("dependencies", result)
        self.assertIn("trajectories", result)


if __name__ == "__main__":
    unittest.main()
