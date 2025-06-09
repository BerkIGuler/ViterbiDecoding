import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.input_source import RandomBinarySource
from src.conv_encoder import ConvolutionalEncoder
from src.viterbi_decoder import ViterbiDecoder
from src.channels import BSCProbabilityChannel

from src.puncturing import (
    create_puncturing_matrix,
    apply_puncturing,
    insert_erasures
)


def simulate_mother_code(information_bits, bit_error_prob):
    """Simulate transmission using the mother (4,1,4) convolutional code.

    Args:
        information_bits (list): Information bits to transmit.
        bit_error_prob (float): BSC bit error probability.

    Returns:
        tuple: (decoded_bits, encoded_length, channel_errors, decoding_errors)
    """
    if isinstance(information_bits, np.ndarray):
        information_bits = information_bits.tolist()

    block_length = len(information_bits)

    encoder = ConvolutionalEncoder(block_length)
    encoded_bits = encoder.encode(information_bits)

    channel = BSCProbabilityChannel(bit_error_prob)
    received_bits = channel.transmit(encoded_bits)

    decoder = ViterbiDecoder(block_length)
    decoded_bits = decoder.decode(received_bits)

    channel_errors = int(np.sum(np.array(encoded_bits) != np.array(received_bits)))
    decoding_errors = int(np.sum(np.array(information_bits) != np.array(decoded_bits)))

    return decoded_bits, len(encoded_bits), channel_errors, decoding_errors


def simulate_punctured_code(information_bits, bit_error_prob):
    """Simulate transmission using the punctured convolutional code.

    Args:
        information_bits (list): Information bits to transmit.
        bit_error_prob (float): BSC bit error probability.

    Returns:
        tuple: (decoded_bits, punctured_length, channel_errors, decoding_errors)
    """
    if isinstance(information_bits, np.ndarray):
        information_bits = information_bits.tolist()

    block_length = len(information_bits)
    puncturing_matrix = create_puncturing_matrix()

    encoder = ConvolutionalEncoder(block_length)
    encoded_bits = encoder.encode(information_bits)

    punctured_bits = apply_puncturing(encoded_bits, puncturing_matrix)

    channel = BSCProbabilityChannel(bit_error_prob)
    received_punctured = channel.transmit(punctured_bits)

    total_time_steps = len(encoded_bits) // 4
    received_with_erasures = insert_erasures(received_punctured, puncturing_matrix, total_time_steps)

    decoder = ViterbiDecoder(block_length)
    decoded_bits = decoder.decode(received_with_erasures, use_punctured=True)

    channel_errors = int(np.sum(np.array(punctured_bits) != np.array(received_punctured)))
    decoding_errors = int(np.sum(np.array(information_bits) != np.array(decoded_bits)))

    return decoded_bits, len(punctured_bits), channel_errors, decoding_errors


def run_comprehensive_ber_comparison(bit_error_probs, num_trials=250, bits_per_trial=200, seed=42):
    """Run comprehensive BER comparison between mother and punctured codes.

    Args:
        bit_error_probs (list): List of BSC bit error probabilities to test.
        num_trials (int): Number of Monte Carlo trials per probability.
        bits_per_trial (int): Number of information bits per trial.
        seed (int): Random seed for reproducible results.

    Returns:
        tuple: (prob_values, mother_ber, punctured_ber, mother_stats, punctured_stats)
    """
    np.random.seed(seed)
    source = RandomBinarySource(p_1=0.5, seed=seed)

    mother_ber_results = []
    punctured_ber_results = []
    mother_statistics = []
    punctured_statistics = []

    for p_error in tqdm(bit_error_probs, desc="BER Analysis"):

        mother_total_errors = 0
        punctured_total_errors = 0
        mother_total_bits = 0
        punctured_total_bits = 0
        mother_channel_errors = 0
        punctured_channel_errors = 0
        mother_transmitted_bits = 0
        punctured_transmitted_bits = 0

        for trial in tqdm(range(num_trials), desc=f"p={p_error:.4f}", leave=False):
            data_bits = source.generate(bits_per_trial)

            decoded_mother, encoded_len, ch_err_m, dec_err_m = simulate_mother_code(
                data_bits, p_error
            )
            mother_total_errors += dec_err_m
            mother_total_bits += len(data_bits)
            mother_channel_errors += ch_err_m
            mother_transmitted_bits += encoded_len

            decoded_punctured, punctured_len, ch_err_p, dec_err_p = simulate_punctured_code(
                data_bits, p_error
            )
            punctured_total_errors += dec_err_p
            punctured_total_bits += len(data_bits)
            punctured_channel_errors += ch_err_p
            punctured_transmitted_bits += punctured_len

        mother_ber = mother_total_errors / mother_total_bits
        punctured_ber = punctured_total_errors / punctured_total_bits

        mother_ber_results.append(mother_ber)
        punctured_ber_results.append(punctured_ber)

        mother_channel_ber = mother_channel_errors / mother_transmitted_bits
        punctured_channel_ber = punctured_channel_errors / punctured_transmitted_bits

        mother_stats = {
            'end_to_end_ber': mother_ber,
            'channel_ber': mother_channel_ber,
            'rate': 1 / 4,
            'transmitted_bits_per_info': encoded_len / bits_per_trial
        }

        punctured_stats = {
            'end_to_end_ber': punctured_ber,
            'channel_ber': punctured_channel_ber,
            'rate': 8 / 20,
            'transmitted_bits_per_info': punctured_len / bits_per_trial
        }

        mother_statistics.append(mother_stats)
        punctured_statistics.append(punctured_stats)

    return (bit_error_probs, mother_ber_results, punctured_ber_results,
            mother_statistics, punctured_statistics)


def plot_ber_vs_channel_probability(prob_values, mother_ber, punctured_ber, save_plot=True):
    """Create BER vs channel probability plot only.

    Args:
        prob_values (list): Channel bit error probabilities.
        mother_ber (list): End-to-end BER for mother code.
        punctured_ber (list): End-to-end BER for punctured code.
        save_plot (bool): Whether to save the plot to file.
    """
    plt.figure(figsize=(10, 7))

    plt.loglog(prob_values, mother_ber, 'ro-', linewidth=2, markersize=8,
               label='Mother Code (4,1,4)', markerfacecolor='white', markeredgewidth=2)
    plt.loglog(prob_values, punctured_ber, 'bs-', linewidth=2, markersize=8,
               label='Punctured Code (Rate 2/5)', markerfacecolor='white', markeredgewidth=2)
    plt.loglog(prob_values, prob_values, 'k--', alpha=0.5, label='Uncoded (p = BER)')

    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('Channel Bit Error Probability (p)', fontsize=12, fontweight='bold')
    plt.ylabel('End-to-End Bit Error Rate', fontsize=12, fontweight='bold')
    plt.title('BER Performance vs Channel Error Probability', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.xlim(min(prob_values), max(prob_values))
    plt.ylim(1e-12, 1)

    plt.tight_layout()

    if save_plot:
        plt.savefig('ber_vs_channel_probability.png', dpi=300, bbox_inches='tight')

    plt.show()


def main():
    """Main function to run the complete comparison analysis."""
    bit_error_probabilities = np.logspace(-2, -0.4, 15)
    num_trials = 10
    bits_per_trial = 10000

    # Run analysis
    results = run_comprehensive_ber_comparison(
        bit_error_probabilities,
        num_trials=num_trials,
        bits_per_trial=bits_per_trial,
        seed=42
    )

    prob_values, mother_ber, punctured_ber, mother_stats, punctured_stats = results

    plot_ber_vs_channel_probability(
        prob_values, mother_ber, punctured_ber
    )


if __name__ == "__main__":
    main()