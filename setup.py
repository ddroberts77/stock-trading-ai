from setuptools import setup, find_packages

setup(
    name="stock-trading-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.1",
        ],
    },
    python_requires=">=3.9",
)