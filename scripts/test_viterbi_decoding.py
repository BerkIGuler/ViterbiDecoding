import numpy as np
from src.conv_encoder import ConvolutionalEncoder
from src.viterbi_decoder import ViterbiDecoder


def main():
    print("=== Viterbi Decoder Test ===")

    block_length = 8
    test_data = [1, 0, 1, 1, 0, 0, 1, 0]

    # Encode
    encoder = ConvolutionalEncoder(block_length)
    encoded_bits = encoder.encode(test_data)

    print(f"Original data:    {test_data}")
    print(f"Encoded bits:     {encoded_bits}")
    print(f"Encoded length:   {len(encoded_bits)}")

    decoder = ViterbiDecoder(block_length)
    decoded_bits = decoder.decode(encoded_bits)

    print(f"Decoded bits:     {list(decoded_bits)}")
    print(f"Decoding correct: {np.array_equal(test_data, decoded_bits)}")

    # Test with single bit error
    print(f"\n=== Test with Single Bit Error ===")
    corrupted_bits = encoded_bits.copy()
    corrupted_bits[5] = 1 - corrupted_bits[5]  # Flip bit at position 5

    print(f"Error at position 5: {encoded_bits[5]} -> {corrupted_bits[5]}")

    decoded_with_error = decoder.decode(corrupted_bits)
    print(f"Decoded with error: {list(decoded_with_error)}")
    print(f"Correction success: {np.array_equal(test_data, decoded_with_error)}")

    print(f"\n=== Trellis Information ===")
    trellis_info = decoder.get_trellis_info()
    for key, value in trellis_info.items():
        print(f"{key}: {value}")

    # Test error correction capability
    print(f"\n=== Error Correction Test ===")
    num_tests = 10
    max_correctable_errors = 0

    for num_errors in range(1, 20):
        corrections = 0
        for test in range(num_tests):
            # Introduce random errors
            test_corrupted = encoded_bits.copy()
            error_positions = np.random.choice(len(encoded_bits), num_errors, replace=False)
            for pos in error_positions:
                test_corrupted[pos] = 1 - test_corrupted[pos]

            # Try to decode
            try:
                test_decoded = decoder.decode(test_corrupted)
                if np.array_equal(test_data, test_decoded):
                    corrections += 1
            except:
                pass

        success_rate = corrections / num_tests
        print(f"{num_errors} errors: {corrections}/{num_tests} corrected ({success_rate:.1%})")

        if success_rate > 0.5:  # If more than half are corrected
            max_correctable_errors = num_errors

    print(f"Estimated error correction capability: ~{max_correctable_errors} errors")


if __name__ == "__main__":
    main()
