import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class PowerPair:
    """Class to store information about a pair of powers"""
    power2: int  # Power of 2
    power3: int  # Power of 3
    value2: int  # 2^power2
    value3: int  # 3^power3
    gap: int     # Absolute difference
    
    @property
    def relative_gap(self) -> float:
        """Calculate relative gap size"""
        return self.gap / max(self.value2, self.value3)

def test_power_pair_creation():
    """Test PowerPair class creation and properties"""
    pair = PowerPair(power2=4, power3=3, value2=16, value3=27, gap=11)
    assert pair.relative_gap == 11/27, "Relative gap calculation incorrect"
    assert pair.gap == 11, "Absolute gap incorrect"

def test_find_closest_pairs():
    """Test finding closest pairs of powers"""
    # Test with small powers to verify manually
    powers_2 = [2**n for n in range(5)]  # [1, 2, 4, 8, 16]
    powers_3 = [3**n for n in range(4)]  # [1, 3, 9, 27]
    
    # Known close pairs for small numbers
    pairs = []
    for p2 in powers_2:
        for p3 in powers_3:
            gap = abs(p2 - p3)
            if gap > 0:
                pairs.append((gap, p2, p3))
    
    pairs.sort()
    assert pairs[0][0] == 1, "First gap should be 1 (between 2 and 3)"
    assert pairs[1][0] == 1, "Second gap should be 1 (between 8 and 9)"

def test_bakers_bound_visualization():
    """Test the visualization code runs without errors"""
    try:
        # Create test data
        pairs = [
            PowerPair(power2=1, power3=1, value2=2, value3=3, gap=1),
            PowerPair(power2=3, power3=2, value2=8, value3=9, gap=1),
            PowerPair(power2=4, power3=3, value2=16, value3=27, gap=11)
        ]
        
        # Test plotting
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
        
        # Plot 1: Powers
        powers2 = [p.power2 for p in pairs]
        powers3 = [p.power3 for p in pairs]
        ax1.scatter(powers2, powers3)
        
        # Plot 2: Absolute gaps
        gaps = [p.gap for p in pairs]
        ax2.semilogy(range(len(gaps)), gaps)
        
        # Plot 3: Relative gaps
        relative_gaps = [p.relative_gap for p in pairs]
        ax3.semilogy(range(len(relative_gaps)), relative_gaps)
        
        plt.close()  # Close plot to avoid display
        success = True
    except Exception as e:
        success = False
        print(f"Visualization failed: {str(e)}")
    
    assert success, "Visualization code should run without errors"

if __name__ == "__main__":
    test_power_pair_creation()
    test_find_closest_pairs()
    test_bakers_bound_visualization()
    print("All power gaps visualization tests passed!") 