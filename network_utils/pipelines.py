# -*- coding: utf-8 -*-

"""Implement Pipeline to assemble .workers.Worker to process .images.Image

"""
from collections import OrderedDict
import numpy as np

from .workers import create_worker, Worker, aug_types


class RandomPipeline(Worker):

    def __init__(self, random_prob=0.5):
        super().__init__()
        self.workers = list()
        self.random_workers = list()
        self.worker_priorities = list()
        self.random_worker_priorities = list()
        self.random_prob = random_prob
        self._rand_state = np.random.RandomState()

    def register(self, worker_name):
        if worker_name in aug_types:
            self._register_random(worker_name)
        else:
            priority = len(self.workers) + len(self.random_workers)
            self.worker_priorities.append(priority)
            self.workers.append(worker_name)

    def _register_random(self, worker_name):
        priority = len(self.workers) + len(self.random_workers)
        self.random_worker_priorities.append(priority)
        self.random_workers.append(worker_name)

    def process(self, *images):
        if self._rand_state.rand() < self.random_prob:
            ind = self._rand_state.choice(range(len(self.random_workers)))
            worker_name = self.random_workers[ind]
            priority = self.random_worker_priorities[ind]
            ind_insert = np.searchsorted(self.worker_priorities, priority,
                                         'right')
            print('rand priority', priority, 'insert', ind_insert)
            workers = self.workers.copy()
            workers.insert(ind_insert, worker_name)
        else:
            workers = self.workers
        for worker_name in workers:
            images = create_worker(worker_name).process(*images)
        return images
