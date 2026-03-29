"""
setup.py — GOEN package installation.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    install_requires = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name             = "goen",
    version          = "1.0.0",
    author           = "Your Name",
    author_email     = "you@example.com",
    description      = "GOEN: Geometry-Optimised Epistemic Network for OOD Detection",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/your-org/goen",
    packages         = find_packages(exclude=["tests*", "scripts*", "notebooks*"]),
    python_requires  = ">=3.10",
    install_requires = install_requires,
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points     = {
        "console_scripts": [
            "goen-train=scripts.train_goen:main",
            "goen-predict=scripts.predict:main",
            "goen-ablation=scripts.ablation:main",
            "goen-seeding=scripts.seeding:main",
        ],
    },
)
