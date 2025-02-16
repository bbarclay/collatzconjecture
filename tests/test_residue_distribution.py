import numpy as np
from collections import defaultdict

def test_residue_distribution():
    """
    Test the distribution of residues after applying 3n+1 transformation
    relative to different powers of 2.
    """
    # Test range
    N = 1000000
    # Powers of 2 to test against
    MODULI = [4, 8, 16, 32]
    
    results = {}
    for modulus in MODULI:
        distributions = defaultdict(int)
        theoretical = {i: N/modulus for i in range(modulus)}
        
        # Test odd numbers only as they're the interesting case
        for n in range(1, N, 2):
            # Apply 3n+1
            next_num = 3 * n + 1
            # Get residue
            residue = next_num % modulus
            distributions[residue] += 1
        
        # Calculate chi-square test statistic
        chi_square = sum((distributions[i] - theoretical[i])**2 / theoretical[i] 
                        for i in range(modulus))
        
        # Calculate distribution error
        max_error = max(abs(distributions[i]/N - 1/modulus) 
                       for i in range(modulus))
        
        results[modulus] = {
            'chi_square': chi_square,
            'max_error': max_error,
            'distributions': dict(distributions)
        }
    
    # Additional test for consecutive τ values
    tau_distributions = defaultdict(int)
    for n in range(1, N, 2):
        next_num = 3 * n + 1
        tau = 0
        while next_num % 2 == 0:
            tau += 1
            next_num //= 2
        tau_distributions[tau] += 1
    
    # Check geometric distribution of τ
    max_tau = max(tau_distributions.keys())
    theoretical_tau = {k: N * (1/2)**(k+1) for k in range(max_tau + 1)}
    tau_error = max(abs(tau_distributions[k]/N - (1/2)**(k+1)) 
                   for k in range(max_tau + 1))
    
    return {
        'residue_distributions': results,
        'tau_distribution': dict(tau_distributions),
        'tau_error': tau_error
    }

if __name__ == "__main__":
    results = test_residue_distribution()
    print("\nResidue Distribution Results:")
    for modulus, data in results['residue_distributions'].items():
        print(f"\nModulo {modulus}:")
        print(f"Chi-square: {data['chi_square']:.2f}")
        print(f"Max error: {data['max_error']:.6f}")
    
    print("\nTau Distribution Error:", results['tau_error'])
    print("\nFirst few tau frequencies:")
    tau_dist = results['tau_distribution']
    for k in sorted(tau_dist.keys())[:5]:
        print(f"τ={k}: {tau_dist[k]}") 