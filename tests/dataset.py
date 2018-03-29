#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt

from network_utils.datasets import Dataset3d

dataset = Dataset3d.from_directory('data', ['image', )
