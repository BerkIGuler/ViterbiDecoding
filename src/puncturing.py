import numpy as np


def create_puncturing_matrix():
    """
    Create the specified puncturing matrix:
    1111 1111
    1111 1111
    1100 1100
    0000 0000

    Returns:
    numpy.ndarray: 4x8 puncturing matrix
    """
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, 1],  # g1: always keep
        [1, 1, 1, 1, 1, 1, 1, 1],  # g2: always keep
        [1, 1, 0, 0, 1, 1, 0, 0],  # g3: keep at times 0,1,4,5
        [0, 0, 0, 0, 0, 0, 0, 0],  # g4: always puncture
    ])


def apply_puncturing(encoded_bits, puncturing_matrix):
    """
    Apply puncturing pattern to encoded bits.

    Parameters:
    encoded_bits (list): Full encoded bits from original encoder
    puncturing_matrix (numpy.ndarray): 4x8 puncturing pattern

    Returns:
    list: Punctured bits (with some bits removed)
    """
    n_generators = 4
    period = puncturing_matrix.shape[1]  # 8

    # Reshape into time steps of 4 bits each
    total_time_steps = len(encoded_bits) // n_generators
    encoded_blocks = np.array(encoded_bits).reshape(total_time_steps, n_generators)

    punctured_bits = []

    for t in range(total_time_steps):
        time_index = t % period  # Cycle through puncturing pattern
        block = encoded_blocks[t]

        # For each generator, check if bit should be kept
        for gen_idx in range(n_generators):
            if puncturing_matrix[gen_idx, time_index] == 1:
                punctured_bits.append(block[gen_idx])

    return punctured_bits


def insert_erasures(received_punctured_bits, puncturing_matrix, total_time_steps):
    """
    Insert erasures (neutral symbols) for punctured positions.

    Parameters:
    received_punctured_bits (list): Received punctured bits
    puncturing_matrix (numpy.ndarray): 4x8 puncturing pattern
    total_time_steps (int): Total number of time steps in original sequence

    Returns:
    list: Full-length sequence with erasures at punctured positions
    """
    n_generators = 4
    period = puncturing_matrix.shape[1]  # 8

    depunctured_bits = []
    punctured_index = 0

    for t in range(total_time_steps):
        time_index = t % period

        for gen_idx in range(n_generators):
            if puncturing_matrix[gen_idx, time_index] == 1:
                # Bit was transmitted - use received bit
                if punctured_index < len(received_punctured_bits):
                    depunctured_bits.append(received_punctured_bits[punctured_index])
                    punctured_index += 1
                else:
                    depunctured_bits.append(0)  # Default if we run out
            else:
                # Bit was punctured - insert erasure (neutral value)
                depunctured_bits.append(0)  # Erasure symbol

    return depunctured_bits