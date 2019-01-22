#!/usr/bin/env python
# -*- coding: utf-8 -*-


class AugmentationStrategy:
    def augment(self, image, label=None, mask=None):
        raise NotImplementedError


class RotationStrategy(AugmentationStrategy):
    def augment(self, image, label=None, mask=None):
        pass


class TranslationStrategy(AugmentationStrategy):
    def augment(self, image, label=None, mask=None):
        pass


class FlippingStrategy(AugmentationStrategy):
    def augment(self, image, label=None, mask=None):
        pass


class DeformationStrategy(AugmentationStrategy):
    def augment(self, image, label=None, mask=None):
        pass


class ScalingStrategy(AugmentationStrategy):
    def augment(self, image, label=None, mask=None):
        pass
