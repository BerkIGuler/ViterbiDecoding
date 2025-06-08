import numpy as np


class PAMModulator:
    def __init__(self, amplitude=1.0):
        """
        PAM (Pulse Amplitude Modulation) modulator.
        Maps 0 -> -a, 1 -> +a

        Parameters:
        amplitude (float): Amplitude 'a' for modulation (default: 1.0)
        """
        self.amplitude = amplitude
        self.signal_power = amplitude ** 2

    def modulate(self, bits):
        """
        Modulate binary bits to PAM symbols.

        Parameters:
        bits (list): List of binary bits (0s and 1s)

        Returns:
        numpy.ndarray: Modulated symbols
        """
        bits = np.array(bits)
        symbols = 2 * bits - 1  # This maps 0->-1, 1->1
        symbols = symbols * self.amplitude
        return symbols

    def get_signal_power(self):
        """Return the signal power (a^2)"""
        return self.signal_power
