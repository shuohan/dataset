# -*- coding: utf-8 -*-

from distutils.core import setup
from glob import glob
import subprocess

command = ['git', 'describe', '--tags']
version = subprocess.check_output(command).decode().strip()

setup(name='dataset',
      version=version,
      description=('Dataset for 3D images with data augmentation and other '
                   'operations'),
      author='Shuo Han',
      author_email='shan50@jhu.edu',
      packages=['dataset'])
