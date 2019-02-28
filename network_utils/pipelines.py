# -*- coding: utf-8 -*-

"""Implement Pipeline to assemble .workers.Worker to process .images.Image

"""
from collections import OrderedDict

from .workers import create_worker, Worker


class Pipeline(Worker):

    def __init__(self):
        self.workers = OrderedDict()

    def register(self, worker_name):
        self.workers[worker_name] = create_worker(worker_name)

    def process(self, *images):
        raise NotImplementedError


class RandomPipeline(Pipeline):

    def __init__(self):
        super().__init__()
        self._rand_state = np.random.RandomState()

    def process(self, *images):
        worker_name = self._rand_state.choice(self.workers.keys())
        images = self.workers[worker_name].process(*images)
        return images


class SerialPipeline(Pipeline):

    def process(self, *images):
        for name, worker in self.workers.items():
            images = worker.process(*images)
        return images
