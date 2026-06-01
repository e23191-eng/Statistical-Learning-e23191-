from setuptools import setup, find_packages

setup(
    name="data_inspector",
    version="0.1.0",
    description="An automated framework for CSV data ingestion, sanitization, and visualization in Colab.",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "plotly>=5.10.0",
        "scipy>=1.9.0",
        "statsmodels>=0.13.0"
    ],
    python_requires=">=3.8",
)