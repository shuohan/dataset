# -*- coding: utf-8 -*-

"""Implement Pipeline to assemble .workers.Worker to process .images.Image

"""
from collections import OrderedDict
import numpy as np
from heapq import heappush, heappop

from .workers import Worker, WorkerName, WorkerType, WorkerTypeMapping
from .workers import create_worker


class RandomPipeline(Worker):
    """Pipeline to select workers to process the images

    The workers in the self._fixed_workers will be always used while only one of
    the workers in self._random_workers will be selected with probability
    self.random_prob (or none of the random workers will be selected). The order
    of the workers are determined by the registration order, e.g. if fixed1,
    random1, random2, fixed2 are inserted, the selected workers will be fixed1,
    none/random1/random2, fixed2

    Attributes:
        random_prob (float): The probability of selecting one of the random
            workers in self._random_workers
        _fixed_workers (list): The list of workers that will be always selected.
            Items are tuple of (priority (int), enum .workers.WorkerName).
        _random_workers (list): The list of workers selected randomly.
            Items are tuple of (priority (int), enum .workers.WorkerName).
        _rand_state (numpy.random.RandomState): Numpy random state

    """
    def __init__(self, random_prob=1):
        super().__init__()
        self.random_prob = random_prob
        self._fixed_workers = list()
        self._random_workers = list()
        self._rand_state = np.random.RandomState()

    def register(self, worker_name):
        """Register worker in to pool for selection

        Args:
            worker_name (str): The name of the worker. It should be in
                .workers.WorkerName.__members__

        Raises:
            RuntimeError: The worker is not in .workers.WorkerName enum

        """
        mapping = WorkerTypeMapping()
        worker_name = WorkerName[worker_name]
        priority = len(self._fixed_workers) + len(self._random_workers)
        if mapping[worker_name] is WorkerType.aug:
            self._random_workers.append((priority, worker_name))
        elif mapping[worker_name] is WorkerType.addon:
            self._fixed_workers.append((priority, worker_name))
        else:
            raise RuntimeError('Worker "%s" does not exist.' % worker_name)

    def process(self, *images):
        """Process a set of .images.Image instances

        Args:
            image (.images.Image): The image to process

        Returns:
            results (tuple of .images.Image): The processed images

        """
        workers = self._fixed_workers.copy()
        if self._rand_state.rand() <= self.random_prob:
            ind = self._rand_state.choice(range(len(self._random_workers)))
            workers.append(self._random_workers[ind])
        for priority, worker_name in sorted(workers):
            images = create_worker(worker_name).process(*images)
        return images
