# -*- coding: utf-8 -*-

from distutils.core import setup
from glob import glob
import subprocess

command = ['git', 'describe', '--tags']
version = subprocess.check_output(command).decode().strip()

setup(name='network-utils',
      version=version,
      description='Deep network utility',
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      packages=['network_util'])
