#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import Config

config = Config()
print(config)
assert config.image_suffixes == ['image']
config.load('input.json')
assert Config().image_suffixes == ['image', 'hello']
config.save('output.json')
