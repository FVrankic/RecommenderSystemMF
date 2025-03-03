import sys
import os

import numpy as np

import time
import logging


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LOG_DIR = os.path.join(BASE_DIR, "logs")

os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "dataset.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=LOG_FILE,
    filemode="a"
)

logger = logging.getLogger(__name__)


class MovieLensDataset:

    """
    A class for loading and processing the MovieLens dataset.
    """

    def __init__(self) -> None:
        """
        Initialize an empty dataset instance.
        """
        self.data: np.ndarray | None = None
        self.name = None


    def load100k(self, path: str = 'datasets/ml-100k/u.data') -> None:

        """
        Load the 100k MovieLens dataset from a file.

        Args:
            path (str, optional): Path to the 100k dataset file. Defaults to 'datasets/ml-100k/u.data'.

        Returns:
            None
        """

        path = os.path.join(BASE_DIR, 'datasets', 'ml-100k', 'u.data')

        start_time = time.time()

        self.data = np.loadtxt(path, delimiter='\t', dtype=int)
        self.name = '100k'

        # Adjust user and item IDs to be 0-based
        self.data[:, 0] -= 1
        self.data[:, 1] -= 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Loaded 100k dataset in {elapsed_time:.2f} seconds.")
        return
    

    def load1m(self, path: str = 'datasets/ml-1m/ratings.dat') -> None:

        """
        Load the 1m MovieLens dataset from a file.

        Args:
            path (str, optional): Path to the 1m dataset file. Defaults to 'datasets/ml-1m/ratings.dat'.

        Returns:
            None
        """

        path = os.path.join(BASE_DIR, 'datasets', 'ml-1m', 'ratings.dat')

        start_time = time.time()

        self.data = np.genfromtxt(path, delimiter="::", dtype=int, autostrip=True)
        self.name = '1m'

        # Adjust user and item IDs to be 0-based
        self.data[:, 0] -= 1
        self.data[:, 1] -= 1

        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Loaded 1m dataset in {elapsed_time:.2f} seconds.")
        return