from setuptools import setup, find_packages

setup(
    name="collatzcrypto",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "seaborn>=0.11.0",
    ],
)
