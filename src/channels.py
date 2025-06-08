import numpy as np


class BSC:
    def __init__(self, snr_db, amplitude=1.0):
        """
        Binary Symmetric Channel with AWGN (Additive White Gaussian Noise).

        Parameters:
        snr_db (float): Signal-to-Noise Ratio in dB
        amplitude (float): Signal amplitude 'a' (default: 1.0)
        """
        self.snr_db = snr_db
        self.amplitude = amplitude
        self.signal_power = amplitude ** 2

        # SNR from dB to linear scale
        self.snr_linear = 10 ** (snr_db / 10)

        self.noise_power = self.signal_power / self.snr_linear
        self.noise_std = np.sqrt(self.noise_power)

    def add_noise(self, symbols):
        """
        Add Gaussian noise to the symbols.

        Parameters:
        symbols (numpy.ndarray): Input symbols

        Returns:
        numpy.ndarray: Noisy symbols
        """
        noise = np.random.normal(0, self.noise_std, len(symbols))
        return symbols + noise

    def hard_decision(self, noisy_symbols):
        """
        Make hard decisions on noisy symbols.
        Decision threshold is 0 (midpoint between -a and +a).

        Parameters:
        noisy_symbols (numpy.ndarray): Noisy received symbols

        Returns:
        numpy.ndarray: Decoded bits (0s and 1s)
        """
        # Threshold at 0: if > 0 -> 1, if <= 0 -> 0
        return (noisy_symbols > 0).astype(int)

    def transmit(self, symbols):
        """
        Complete transmission: add noise and make hard decisions.

        Parameters:
        symbols (numpy.ndarray): Input symbols

        Returns:
        tuple: (noisy_symbols, decoded_bits)
        """
        noisy_symbols = self.add_noise(symbols)
        decoded_bits = self.hard_decision(noisy_symbols)
        return noisy_symbols, decoded_bits

    def get_channel_parameters(self):
        """Return channel parameters"""
        return {
            'snr_db': self.snr_db,
            'snr_linear': self.snr_linear,
            'signal_power': self.signal_power,
            'noise_power': self.noise_power,
            'noise_std': self.noise_std
        }
