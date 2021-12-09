#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="nw3",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "Flask",
        "fire",
        "jsonlines",
    ],
    zip_safe=False,
)
