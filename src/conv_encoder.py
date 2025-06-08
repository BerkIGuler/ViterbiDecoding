class ConvolutionalEncoder:
    def __init__(self, block_length):
        """
        Initialize the convolutional encoder for rate 1/4 code.

        Parameters:
        block_length (int): Number of information bits to encode
        """
        self.block_length = block_length
        self.memory_length = 4
        self.rate_k = 1
        self.rate_n = 4


        self.generators = [
            [1, 0, 0, 1, 1],  # g(1)
            [1, 1, 1, 0, 1],  # g(2)
            [1, 0, 1, 1, 1],  # g(3)
            [1, 1, 0, 1, 1]  # g(4)
        ]

        # Initialize shift register to all zeros
        self.shift_register = [0] * self.memory_length

    def reset(self):
        """clear shift registers"""
        self.shift_register = [0] * self.memory_length

    def encode_bit(self, input_bit):
        """
        Encode a single input bit and return n output bits.

        Parameters:
        input_bit (int): Input bit (0 or 1)

        Returns:
        list: List of n output bits
        """
        current_state = [input_bit] + self.shift_register

        # Generate output bits using each generator
        output_bits = []
        for generator in self.generators:
            output_bit = 0
            # Compute convolution for this generator
            for i in range(len(generator)):
                # & is bitwise AND,
                # ^ is bit-wise XOR in Python
                output_bit ^= (current_state[i] & generator[i])
            output_bits.append(output_bit)

        # shift right and insert new input
        self.shift_register = [input_bit] + self.shift_register[:-1]

        return output_bits

    def encode(self, information_bits):
        """
        Encode a sequence of information bits.
        Appends m=4 termination bits (0000) to reset encoder to state 0000.

        Parameters:
        information_bits (list): List of information bits

        Returns:
        list: Encoded bits
        """
        if len(information_bits) != self.block_length:
            raise ValueError(f"Expected {self.block_length} information bits, got {len(information_bits)}")

        # Reset encoder state
        self.reset()

        # Append termination bits (m zeros) to input sequence
        termination_bits = [0] * self.memory_length  # [0, 0, 0, 0]
        extended_input = information_bits + termination_bits

        encoded_bits = []

        # Encode all bits (information + termination)
        for bit in extended_input:
            output_bits = self.encode_bit(bit)
            encoded_bits.extend(output_bits)

        return encoded_bits

    def get_code_parameters(self):
        """Return the code parameters (n, k, m)"""
        return self.rate_n, self.rate_k, self.memory_length

    def get_total_encoded_length(self):
        """Return the total length of encoded sequence including termination"""
        return (self.block_length + self.memory_length) * self.rate_n


# testing
if __name__ == "__main__":
    # Create encoder for block length of 8 bits
    encoder = ConvolutionalEncoder(block_length=8)

    print(f"Code parameters (n,k,m): {encoder.get_code_parameters()}")
    print(f"Total encoded length for block_length={encoder.block_length}: {encoder.get_total_encoded_length()}")

    # Test with a simple sequence
    info_bits = [1, 0, 1, 1, 0, 0, 1, 0]
    print(f"\nInformation bits: {info_bits}")
    print(f"Termination bits appended: [0, 0, 0, 0]")
    print(f"Total input sequence: {info_bits + [0, 0, 0, 0]}")

    encoded = encoder.encode(info_bits)
    print(f"Encoded bits: {encoded}")
    print(f"Encoded length: {len(encoded)}")

    # Verify the rate
    expected_length = (len(info_bits) + encoder.memory_length) * encoder.rate_n
    print(f"Expected length: {expected_length}")
    print(f"Rate check: {len(info_bits)} info bits -> {len(encoded)} encoded bits")
    print(
        f"Note: During decoding, we'll discard the last {encoder.memory_length * encoder.rate_n} bits (termination bits)")