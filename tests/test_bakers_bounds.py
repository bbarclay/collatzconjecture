import numpy as np
from math import log2, log
import pytest

def find_closest_power_pairs(max_power=30):
    """Find pairs of powers of 2 and 3 that are closest to each other."""
    powers_2 = [2**n for n in range(max_power)]
    powers_3 = [3**n for n in range(max_power)]
    
    gaps = []
    for p2 in powers_2:
        for p3 in powers_3:
            gap = abs(p2 - p3)
            if gap > 0:  # Exclude exact matches (which shouldn't exist)
                gaps.append((gap, p2, p3))
    
    return sorted(gaps)[:10]  # Return 10 closest pairs

def verify_bakers_bound(gap, max_val):
    """Verify that the gap satisfies Baker's bound."""
    # Conservative values for Baker's constants
    C = 0.1
    kappa = 3.0
    
    # Calculate Baker's bound
    theoretical_min = C / (max_val**kappa)
    relative_gap = gap / max_val
    
    return relative_gap > theoretical_min

def test_bakers_bounds_on_collatz():
    """Test that Baker's bounds hold for Collatz trajectories."""
    # Find closest power pairs
    closest_pairs = find_closest_power_pairs()
    
    # Test each pair satisfies Baker's bounds
    for gap, p2, p3 in closest_pairs:
        max_val = max(p2, p3)
        assert verify_bakers_bound(gap, max_val), \
            f"Gap {gap} between {p2} and {p3} violates Baker's bound"
    
    # Additional test: Check relative gaps don't get arbitrarily small
    relative_gaps = [gap/max(p2,p3) for gap, p2, p3 in closest_pairs]
    min_relative_gap = min(relative_gaps)
    
    # The minimum relative gap should not be too small
    assert min_relative_gap > 1e-6, \
        f"Relative gap {min_relative_gap} is suspiciously small"
    
    # Test specific numbers known to have high τ values
    test_numbers = [27, 31, 41, 271, 626331]
    for n in test_numbers:
        trajectory = []
        power2_count = 0
        power3_count = 0
        current = n
        
        # Follow trajectory until we hit 1
        while current != 1 and len(trajectory) < 1000:
            trajectory.append(current)
            if current % 2 == 0:
                current //= 2
                power2_count += 1
            else:
                current = 3 * current + 1
                power3_count += 1
                # Count trailing zeros
                while current % 2 == 0:
                    current //= 2
                    power2_count += 1
        
        # Check power balance
        if power3_count > 0:  # Avoid division by zero
            ratio = power2_count / power3_count
            # log₂(3) ≈ 1.58496... - ratio should be greater
            assert ratio > 1.58496, \
                f"Number {n} has suspicious power ratio {ratio}"

if __name__ == "__main__":
    test_bakers_bounds_on_collatz()
    print("All Baker's bounds tests passed!") 