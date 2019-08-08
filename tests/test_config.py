#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataset import Config

Config.show()
assert Config.image_suffixes == ['image']
Config.load_json('input.json')
assert Config.image_suffixes == ['image', 'hello']
Config.save_json('output.json')
