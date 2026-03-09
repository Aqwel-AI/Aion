#!/usr/bin/env python3
"""
Setup script for Aqwel-Aion v0.1.8
Professional AI Research & Development Library

Author: Aksel Aghajanyan
Developed by: Aqwel AI Team
Copyright: 2025 Aqwel AI
License: Apache-2.0
"""

import os
from setuptools import setup, find_packages, Extension


def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip()
            for line in fh
            if line.strip() and not line.startswith("#")
        ]


def _get_extensions():
    """Build C++ extension if pybind11 and source are available."""
    try:
        import pybind11
        include = [pybind11.get_include()]
    except ImportError:
        return []
    # Use path relative to setup.py directory (no absolute paths)
    src = "src/aion_core.cpp"
    if not os.path.isfile(src):
        return []
    return [
        Extension(
            "aion._aion_core",
            sources=[src],
            include_dirs=include,
            extra_compile_args=["-O3", "-std=c++14"] if os.name != "nt" else ["/O2", "/std:c++14"],
            language="c++",
        )
    ]


setup(
    name="aqwel-aion",
    version="0.1.8",
    author="Aksel Aghajanyan",
    maintainer="Aqwel AI Team",
    description=(
        "Complete AI Research & Development Library with "
        "Advanced Mathematics, AI, and Visualization Tools"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://aqwelai.xyz/",
    project_urls={
        "Homepage": "https://aqwelai.xyz",
        "Documentation": "https://aqwelai.xyz/#/docs",
        "PyPI": "https://pypi.org/project/aqwel-aion/",
    },
    packages=find_packages(),
    ext_modules=_get_extensions(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Text Processing",
        "Topic :: Utilities",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords=[
        "ai-research",
        "machine-learning",
        "mathematics",
        "statistics",
        "data-science",
        "visualization",
        "scientific-computing",
        "research-tools",
        "aqwel-ai",
    ],
    python_requires=">=3.8",

    # Core dependencies required for all users
    install_requires=[
        "numpy>=1.21.0",
        "watchdog>=2.1.0",
        "gitpython>=3.1.0",
    ],

    # Optional feature dependencies
    extras_require={

        # Visualization support
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],

        # Transformer training (Aion Former: decoder-only, NumPy autograd)
        "former": [
            "matplotlib>=3.5.0",
            "pyyaml>=6.0",
        ],

        # Machine learning and AI stack
        "ai": [
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "sentence-transformers>=2.2.0",
        ],

        # Documentation and export tools
        "docs": [
            "reportlab>=3.6.0",
            "pillow>=8.0.0",
        ],

        # Development tools
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],

        # Full installation (all features)
        "full": [
            "scipy>=1.7.0",
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "transformers>=4.20.0",
            "torch>=1.12.0",
            "openai>=1.0.0",
            "faiss-cpu>=1.7.0",
            "sentence-transformers>=2.2.0",
            "reportlab>=3.6.0",
            "pillow>=8.0.0",
        ],
    },

    entry_points={
        "console_scripts": [
            "aion=aion.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    license="Apache-2.0",
)
