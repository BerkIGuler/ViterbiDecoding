import numpy as np

class RandomBinarySource:
    def __init__(self, p_1=0.5, seed=None):
        """
        Random binary source with configurable probability.

        Parameters:
        p_1 (float): Probability of generating bit 1 (default: 0.5)
        seed (int): Random seed for reproducibility (optional)
        """
        if seed is not None:
            np.random.seed(seed)

        self.p_1 = p_1
        self.p_0 = 1 - p_1

    def generate(self, length):
        """
        Generate random binary sequence using numpy.

        Parameters:
        length (int): Number of bits to generate

        Returns:
        numpy.ndarray: Random binary sequence
        """
        random_values = np.random.random(length)
        return (random_values < self.p_1).astype(int)
