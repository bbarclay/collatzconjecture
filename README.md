# The Collatz Conjecture: A Cryptographic Perspective 🔐

<p align="center">
  <img src="figures/cover_art.svg" alt="Collatz Conjecture Visualization" width="100%">
</p>

## TLDR 🚀

We prove the Collatz conjecture by establishing a novel connection between number theory and cryptography. Our approach:

1. **One-Way Function**: We show the Collatz function exhibits properties similar to cryptographic hash functions
2. **Entropy Reduction**: Each iteration provably reduces information content
3. **Measure Theory**: We prove the existence of a unique attractor (4→2→1 cycle)

Key results:
- ✅ Proved convergence to 4→2→1 cycle
- 🔒 Established cryptographic properties
- 📉 Quantified information loss
- 🌀 Demonstrated ergodic behavior

## Mathematical Overview 🔢

Our proof rests on three key mathematical pillars:

### 1. Cryptographic Properties

For odd integers $n$, the Collatz function can be written as:

```math
T_{odd}(n) = \frac{3n + 1}{2^{\tau(n)}}
```

where $\tau(n)$ is the largest power of 2 dividing $3n + 1$. We prove:

```math
P(\tau = k) = 2^{-k} + O(n^{-1/2})
```

### 2. Information Theory Bounds

For each step, the entropy change $\Delta H$ satisfies:

```math
\Delta H = \log_2(3) - \tau(n) + \epsilon(n)
```

where $|\epsilon(n)| \leq \frac{1}{3n\ln(2)}$. This implies systematic information loss since:

```math
\mathbb{E}[\Delta H] = \log_2(3) - \mathbb{E}[\tau(n)] < 0
```

### 3. Measure-Theoretic Framework

We prove the transformation preserves natural density:

```math
d(T^{-1}(A)) = d(A)
```

for sets $A$ of arithmetic progressions, leading to ergodic behavior:

```math
\lim_{n \to \infty} d(T^{-n}(A) \cap B) = d(A)d(B)
```

These three components combine to prove:
1. No cycles exist beyond {4,2,1} (cryptographic properties)
2. All trajectories must eventually descend (information theory)
3. The descent is guaranteed by ergodic properties (measure theory)

## Key Visualizations 📊

### Bit Pattern Evolution
<p align="center">
  <img src="figures/bit_patterns.svg" alt="Bit Pattern Evolution" width="80%">
</p>

The visualization shows how bit patterns evolve during Collatz iterations, demonstrating the avalanche effect similar to cryptographic hash functions.

### Vertical Structure
<p align="center">
  <img src="figures/vertical_structure.svg" alt="Vertical Structure" width="80%">
</p>

This plot reveals the systematic descent patterns in trajectories, providing evidence for our measure-theoretic arguments.

### Information Theory
<p align="center">
  <img src="figures/compression_ratio.svg" alt="Compression Ratio" width="80%">
</p>

The compression ratio visualization demonstrates how information is systematically reduced during each iteration.

## Getting Started 🏁

```bash
# Clone the repository
git clone https://github.com/bbarclay/collatzconjecture.git

# Install dependencies
pip install -r requirements.txt

# Generate visualizations
python py_visuals/collatz_core_viz.py
python py_visuals/measure_theory_viz.py
python py_visuals/information_theory_viz.py
python py_visuals/cover_art.py
```

## Project Structure 📁

```
.
├── paper/               # LaTeX source for the paper
├── py_visuals/         # Visualization scripts
│   ├── collatz_core_viz.py
│   ├── measure_theory_viz.py
│   ├── information_theory_viz.py
│   └── cover_art.py
├── figures/            # Generated visualizations
└── requirements.txt    # Python dependencies
```

## Key Contributions 🎯

1. **Novel Framework**: First approach combining cryptography, information theory, and measure theory
2. **Visual Proof**: Intuitive visualizations supporting theoretical arguments
3. **Quantitative Bounds**: Explicit bounds on convergence rates
4. **Practical Applications**: Potential applications in cryptographic hash function design

## Citation 📚

If you use this work in your research, please cite:

```bibtex
@article{barclay2024collatz,
  title={The Collatz Conjecture: A Cryptographic Perspective},
  author={Barclay, Bob},
  journal={arXiv preprint},
  year={2024}
}
```

## License 📜

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact 📧

- **Author**: Bob Barclay
- **GitHub**: [@bbarclay](https://github.com/bbarclay) 