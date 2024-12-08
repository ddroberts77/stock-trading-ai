from setuptools import setup, find_packages

setup(
    name="stock-trading-ai",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.24.0',
    ],
)