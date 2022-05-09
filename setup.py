import os
import platform
from setuptools import setup

with open('requirements.txt', 'r') as f:
    dependencies = f.read().splitlines()

with open('dev-requirements.txt', 'r') as f:
    dev_dependencies = f.read().splitlines()

setup(
    name="PacBot",
    author="Erin Aho",
    install_requires=dependencies,
    version="1.0.0",
    extras_require={
        "dev": dev_dependencies,
    },
)
