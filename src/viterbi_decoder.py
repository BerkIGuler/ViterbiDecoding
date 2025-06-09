import numpy as np


class ViterbiDecoder:
    def __init__(self, block_length):
        """
        Viterbi decoder for (4,1,4) convolutional code.

        Parameters:
        block_length (int): Number of information bits (excluding termination)
        """
        self.block_length = block_length
        self.memory_length = 4
        self.rate_k = 1
        self.rate_n = 4
        self.num_states = 2 ** self.memory_length

        self.generators = [
            [1, 0, 0, 1, 1],  # g(1)
            [1, 1, 1, 0, 1],  # g(2)
            [1, 0, 1, 1, 1],  # g(3)
            [1, 1, 0, 1, 1]  # g(4)
        ]

        self._build_trellis()

    def _build_trellis(self):
        """
        Build the trellis structure by computing all state transitions
        and their corresponding output symbols.
        """
        self.next_state = np.zeros((self.num_states, 2), dtype=int)
        self.output_symbols = np.zeros((self.num_states, 2, self.rate_n), dtype=int)

        for state in range(self.num_states):
            # Convert state to binary representation
            state_bits = [(state >> i) & 1 for i in range(self.memory_length - 1, -1, -1)]

            for input_bit in [0, 1]:
                current_state = [input_bit] + state_bits

                output = []
                for generator in self.generators:
                    output_bit = 0
                    for i in range(len(generator)):
                        output_bit ^= (current_state[i] & generator[i])
                    output.append(output_bit)

                next_state_bits = [input_bit] + state_bits[:-1]
                # compute next state in decimal
                next_state = 0
                for i, bit in enumerate(next_state_bits):
                    next_state += bit * (2 ** (self.memory_length - 1 - i))

                self.next_state[state, input_bit] = next_state
                self.output_symbols[state, input_bit] = output

    @staticmethod
    def _hamming_distance(received, expected):
        """
        Compute Hamming distance between received and expected symbols.

        Parameters:
        received (numpy.ndarray): Received symbols (4 bits)
        expected (numpy.ndarray): Expected symbols (4 bits)

        Returns:
        int: Hamming distance
        """
        return int(np.sum(received != expected))

    @staticmethod
    def _hamming_distance_punctured(received, expected):
        """
        Compute Hamming distance between received and expected symbols,
        skipping positions where received[i] is None (punctured positions).

        Parameters:
        received (list or numpy.ndarray): Received symbols (4 bits, some may be None)
        expected (numpy.ndarray): Expected symbols (4 bits)

        Returns:
        int: Hamming distance considering only non-punctured positions
        """
        distance = 0
        for i in range(len(received)):
            if received[i] is not None:  # Skip punctured positions
                if int(received[i]) != int(expected[i]):
                    distance += 1
        return distance

    def decode(self, received_symbols, use_punctured=False):
        """
        Decode received symbols using Viterbi algorithm.

        Parameters:
        received_symbols (list or numpy.ndarray): Received coded symbols
        use_punctured (bool): If True, use punctured decoding (skip None values)

        Returns:
        numpy.ndarray: Decoded information bits
        """
        total_symbols = len(received_symbols)
        if total_symbols % self.rate_n != 0:
            raise ValueError(f"Received symbols length must be multiple of {self.rate_n}")

        time_steps = total_symbols // self.rate_n

        if use_punctured:
            received_blocks = []
            for t in range(time_steps):
                start_idx = t * self.rate_n
                end_idx = start_idx + self.rate_n
                received_blocks.append(received_symbols[start_idx:end_idx])
        else:
            received_symbols = np.array(received_symbols)
            received_blocks = received_symbols.reshape(time_steps, self.rate_n)


        path_metrics = np.full((time_steps + 1, self.num_states), np.inf)
        path_metrics[0, 0] = 0  # Start from state 0

        survivor_paths = np.zeros((time_steps, self.num_states, 2), dtype=int)

        for t in range(time_steps):
            received_block = received_blocks[t]

            for current_state in range(self.num_states):
                if path_metrics[t, current_state] == np.inf:
                    continue

                # Try both possible input bits
                for input_bit in [0, 1]:
                    next_state = self.next_state[current_state, input_bit]
                    expected_output = self.output_symbols[current_state, input_bit]

                    if use_punctured:
                        branch_metric = self._hamming_distance_punctured(received_block, expected_output)
                    else:
                        branch_metric = self._hamming_distance(received_block, expected_output)

                    new_metric = path_metrics[t, current_state] + branch_metric

                    if new_metric < path_metrics[t + 1, next_state]:
                        path_metrics[t + 1, next_state] = new_metric
                        survivor_paths[t, next_state] = [current_state, input_bit]

        final_state = int(np.argmin(path_metrics[time_steps]))

        decoded_bits = []
        current_state = final_state

        for t in range(time_steps - 1, -1, -1):
            prev_state, input_bit = survivor_paths[t, current_state]
            decoded_bits.append(int(input_bit))
            current_state = int(prev_state)

        decoded_bits.reverse()

        # Remove termination bits (last m bits)
        information_bits = decoded_bits[:self.block_length]

        return np.array(information_bits)

    def get_trellis_info(self):
        """Return trellis structure information for debugging."""
        return {
            'num_states': self.num_states,
            'next_state_shape': self.next_state.shape,
            'output_symbols_shape': self.output_symbols.shape
        }