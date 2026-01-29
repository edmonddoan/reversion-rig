"""
Setup configuration for reversion-rig package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="reversion-rig",
    version="1.0.0",
    author="reversion-rig",
    description="Mean Reversion Trading Strategy with Backtesting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reversion-rig/reversion-rig",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "yfinance>=0.2.0",
        "matplotlib>=3.5.0",
        "plotly>=5.10.0",
        "python-dateutil>=2.8.0",
        "pytz>=2022.1",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    keywords=[
        "trading",
        "strategy",
        "mean-reversion",
        "backtesting",
        "finance",
        "investing",
        "algorithmic-trading",
        "technical-analysis",
    ],
)
