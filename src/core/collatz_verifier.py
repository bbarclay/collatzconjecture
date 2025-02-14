"""
Core verification tools for the Collatz conjecture proof.
This module implements the fundamental algorithms and verification methods.
"""

import math
from typing import List, Tuple, Dict, Optional


class CollatzVerifier:
    def __init__(self, max_steps=1000000):
        self.max_steps = max_steps
        self.cache = {}  # For memoization

    def collatz_step(self, n: int) -> int:
        """Single step of Collatz transformation"""
        return n // 2 if n % 2 == 0 else 3 * n + 1

    def trajectory(self, start: int) -> Tuple[List[int], bool]:
        """Compute full trajectory until 1 or max_steps"""
        n = start
        path = [n]
        steps = 0
        while n != 1 and steps < self.max_steps:
            n = self.collatz_step(n)
            path.append(n)
            steps += 1
        return path, steps == self.max_steps

    def find_tau(self, n: int) -> int:
        """
        Compute τ(n) for odd n, where τ(n) is the number of times we can divide
        3n + 1 by 2 before getting an odd number
        """
        if n % 2 == 0:
            raise ValueError("n must be odd")

        # First apply 3n + 1
        x = 3 * n + 1
        tau = 0

        # Then count divisions by 2 until we get an odd number
        while x % 2 == 0:
            tau += 1
            x //= 2
        return tau

    def verify_no_even_cycles(self, limit: int = 10000) -> Tuple[bool, Optional[int]]:
        """Verify no even cycles exist above 4"""
        for n in range(6, limit + 1, 2):
            path, _ = self.trajectory(n)
            evens = [x for x in path if x % 2 == 0 and x > 4]
            if len(set(evens)) != len(evens):
                return False, n  # Found a cycle
        return True, None

    def verify_forward_uniqueness(
        self, limit: int = 10000
    ) -> Tuple[bool, Optional[int]]:
        """Verify each odd n has unique successor"""
        successors = {}
        for n in range(1, limit + 1, 2):
            x = 3 * n + 1
            while x % 2 == 0:
                x //= 2
            if n in successors and successors[n] != x:
                return False, n
            successors[n] = x
        return True, None

    def analyze_tau_statistics(
        self, limit: int = 100000
    ) -> Tuple[Dict[int, float], float]:
        """Analyze statistical properties of τ"""
        tau_counts = {}
        total_tau = 0
        for n in range(1, limit + 1, 2):
            tau = self.find_tau(n)
            tau_counts[tau] = tau_counts.get(tau, 0) + 1
            total_tau += tau

        # Calculate empirical probabilities
        total = sum(tau_counts.values())
        probs = {k: v / total for k, v in tau_counts.items()}
        avg_tau = total_tau / total

        return probs, avg_tau

    def measure_entropy_changes(
        self, start: int, max_steps: int = 100
    ) -> List[Tuple[float, float, float]]:
        """Measure entropy changes in trajectory"""

        def binary_entropy(x):
            return math.log2(x)

        path, _ = self.trajectory(start)
        changes = []

        for i in range(len(path) - 1):
            if path[i] % 2 == 1:  # Only measure odd steps
                delta_h = binary_entropy(path[i + 1]) - binary_entropy(path[i])
                tau = self.find_tau(path[i])
                theoretical = math.log2(3) - tau
                error = delta_h - theoretical
                changes.append((delta_h, theoretical, error))

        return changes
