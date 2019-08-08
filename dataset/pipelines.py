# -*- coding: utf-8 -*-

"""Pipelines to process images.

A pipeline assembles multiple :class:`dataset.workers.Worker` in order to
process instances of :class:`dataset.images.Image`.

Todo:
    * Use a queue; only process images if the queue is empty otherwise pop.
    * Add more types of pipelines.

"""
import numpy as np

from .config import Config
from .workers import Worker, WorkerType, WorkerCreator


class RandomPipeline(Worker):
    """Selects one augmentation worker randomly to process the images.

    The workers in the attribute :attr:`fixed_workers` will be always used while
    only one of the workers in the attribute :attr:`random_workers` will be
    selected with probability :attr:`random_prob` (or none of the random workers
    will be selected). The order of the workers are determined by the
    registration order, e.g. if fixed1, random1, random2, fixed2 are inserted,
    the selected workers will be fixed1, none/random1/random2, fixed2.

    Attributes:
        random_prob (float): The probability of selecting one of the random
            workers in the attribute :attr:`random_workers`.
        fixed_workers (list): The list of workers that will be always selected.
            Items are :class:`tuple` of (``priority`` (:class:`int`),
            ``worker_name`` (:class:`str`)). ``priority`` is used to determine
            the processing order.
        random_workers (list): The list of workers selected randomly.
            Items are :class:`tuple` of (``priority`` (:class:`int`),
            ``worker_name`` (:class:`str`)). ``priority`` is not used here since
            at most one random workers will be selected.

    """
    def __init__(self):
        super().__init__()
        self.random_prob = Config().aug_prob
        self.fixed_workers = list()
        self.random_workers = list()
        self._rand_state = np.random.RandomState()

    def register(self, *worker_names):
        """Registers workers for selection.

        Args:
            worker_name (str): The name of the worker. It should be registered
                in the class :class:`dataset.workers.WorkerCreator`.

        Raises:
            RuntimeError: The worker is not registered in the class
                :class:`dataset.workers.WorkerCreator`.

        """
        creator = WorkerCreator()
        for name in worker_names:
            priority = len(self.fixed_workers) + len(self.random_workers)
            if creator.get_type(name) is WorkerType.AUG:
                self.random_workers.append((priority, name))
            elif creator.get_type(name) is WorkerType.ADDON:
                self.fixed_workers.append((priority, name))
            else:
                raise RuntimeError('Worker "%s" does not exist.' % name)

    def process(self, *images):
        """Processes multiple :class:`dataset.images.Image` instances.

        Args:
            image (dataset.images.Image): The image to process.

        Returns:
            tuple[dataset.images.Image]: The processed images.

        """
        workers = self.fixed_workers.copy()
        if self.random_workers and self._rand_state.rand() <= self.random_prob:
            # choice does not support list of tuple
            ind = self._rand_state.choice(range(len(self.random_workers)))
            workers.append(self.random_workers[ind])
        for priority, worker_name in sorted(workers):
            images = WorkerCreator().create(worker_name).process(*images)
        return images
