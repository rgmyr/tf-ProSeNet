#!/usr/bin/env python3
import os
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(name='prosenet',
      version='0.1',
      description='tf.keras implementation of `ProSeNet`.',
      url='https://github.com/rgmyr/tf-ProSeNet',
      author='Ross Meyer',
      author_email='ross.meyer@utexas.edu',
      packages=find_packages(PACKAGE_PATH),
      install_requires=[
            'numpy >= 1.13.0',
      ],
      zip_safe=False
)
