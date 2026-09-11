"""Setup script for the rule-research extract."""

from setuptools import setup, find_packages

import os

# Read README
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
        long_description = "Closed-bar feature research and court"

# Read requirements
requirements = []
if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        requirements = [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]
else:
    # Minimal requirements if file doesn't exist
    requirements = [
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "lightgbm>=3.3.0",
    ]

setup(
    name="interpretable-ml-trading",
    version="0.1.0",
    author="TradeEdgeX",
    author_email="",
    description="Closed-bar features, event backtest, and a research court — not an ML trading bot",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TradeEdgeX/interpretable-ml-trading",
    packages=find_packages(where=".", include=["src*", "scripts*"]),
    package_dir={
        "": ".",
    },
    package_data={
        "src.lab": ["static/*"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mlbot=cli.main:main",
        ],
    },
)
