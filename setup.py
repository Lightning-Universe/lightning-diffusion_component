#!/usr/bin/env python

import os
from importlib.util import module_from_spec, spec_from_file_location

from setuptools import find_packages, setup

_PATH_ROOT = os.path.dirname(__file__)
PACKAGE_NAME = "lightning_diffusion"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name=PACKAGE_NAME,
    version="0.0.1",
    description="",
    author="",
    author_email="",
    url="",
    packages=find_packages(exclude=["tests", "docs"]),
    long_description="",
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["deep learning", "pytorch", "AI"],
    python_requires=">=3.8",
    setup_requires=["wheel"],
    install_requires=requirements,
)
