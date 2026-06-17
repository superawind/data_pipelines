#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from pathlib import Path

# 读取README文件
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# 读取requirements文件
requirements = (this_directory / "requirements.txt").read_text(encoding='utf-8').splitlines()

setup(
    name="ai-verifier",
    version="1.0.0",
    author="AI Verifier Team",
    author_email="dev@ai-verifier.com",
    description="工业级AI回应验证系统 - 混合验证模式",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ai-verifier/ai-verifier",
    project_urls={
        "Bug Reports": "https://github.com/ai-verifier/ai-verifier/issues",
        "Source": "https://github.com/ai-verifier/ai-verifier",
        "Documentation": "https://ai-verifier.readthedocs.io",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
            "pre-commit>=2.20.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-verifier=ai_verifier.cli:main",
            "ai-verify=ai_verifier.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ai_verifier": [
            "prompts/*.txt",
            "config/*.json",
        ],
    },
    zip_safe=False,
    keywords=[
        "ai", "verification", "testing", "quality-assurance", 
        "llm", "code-verification", "automated-testing"
    ],
) 