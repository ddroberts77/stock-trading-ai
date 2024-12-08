from setuptools import setup, find_packages

setup(
    name="stock-trading-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'scikit-learn>=1.3.0',
        'yfinance>=0.2.30',
        'torch>=2.0.0',
        'tensorflow>=2.13.0'
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'matplotlib>=3.7.0',
            'seaborn>=0.12.0'
        ]
    }
)