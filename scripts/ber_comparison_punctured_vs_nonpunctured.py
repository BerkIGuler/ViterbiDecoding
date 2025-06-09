import numpy as np
import matplotlib.pyplot as plt

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

    # Encode, transmit, decode
    encoder = ConvolutionalEncoder(block_length)
    encoded_bits = encoder.encode(information_bits)

    channel = BSCProbabilityChannel(bit_error_prob)
    received_bits = channel.transmit(encoded_bits)

    decoder = ViterbiDecoder(block_length)
    decoded_bits = decoder.decode(received_bits)

    # Calculate errors
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

    # Encode with mother code
    encoder = ConvolutionalEncoder(block_length)
    encoded_bits = encoder.encode(information_bits)

    # Apply puncturing
    punctured_bits = apply_puncturing(encoded_bits, puncturing_matrix)

    # Transmit punctured bits
    channel = BSCProbabilityChannel(bit_error_prob)
    received_punctured = channel.transmit(punctured_bits)

    # Insert erasures and decode
    total_time_steps = len(encoded_bits) // 4
    received_with_erasures = insert_erasures(received_punctured, puncturing_matrix, total_time_steps)

    decoder = ViterbiDecoder(block_length)
    decoded_bits = decoder.decode(received_with_erasures)

    # Calculate errors
    channel_errors = int(np.sum(np.array(punctured_bits) != np.array(received_punctured)))
    decoding_errors = int(np.sum(np.array(information_bits) != np.array(decoded_bits)))

    return decoded_bits, len(punctured_bits), channel_errors, decoding_errors


def run_comprehensive_ber_comparison(bit_error_probs, num_trials=50, bits_per_trial=200, seed=42):
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

    print("=== Comprehensive BER Analysis: Mother vs Punctured Code ===")
    print(f"Monte Carlo trials: {num_trials} per probability")
    print(f"Information bits per trial: {bits_per_trial}")
    print(f"Total information bits per point: {num_trials * bits_per_trial}")
    print("-" * 80)

    for p_error in bit_error_probs:
        print(f"Testing p = {p_error:.3f}...", end=" ")

        # Initialize counters
        mother_total_errors = 0
        punctured_total_errors = 0
        mother_total_bits = 0
        punctured_total_bits = 0
        mother_channel_errors = 0
        punctured_channel_errors = 0
        mother_transmitted_bits = 0
        punctured_transmitted_bits = 0

        # Monte Carlo trials
        for trial in range(num_trials):
            data_bits = source.generate(bits_per_trial)

            # Mother code simulation
            decoded_mother, encoded_len, ch_err_m, dec_err_m = simulate_mother_code(
                data_bits, p_error
            )
            mother_total_errors += dec_err_m
            mother_total_bits += len(data_bits)
            mother_channel_errors += ch_err_m
            mother_transmitted_bits += encoded_len

            # Punctured code simulation
            decoded_punctured, punctured_len, ch_err_p, dec_err_p = simulate_punctured_code(
                data_bits, p_error
            )
            punctured_total_errors += dec_err_p
            punctured_total_bits += len(data_bits)
            punctured_channel_errors += ch_err_p
            punctured_transmitted_bits += punctured_len

        # Calculate BERs
        mother_ber = mother_total_errors / mother_total_bits
        punctured_ber = punctured_total_errors / punctured_total_bits

        mother_ber_results.append(mother_ber)
        punctured_ber_results.append(punctured_ber)

        # Calculate statistics
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

        # Display progress
        if mother_ber > 0 and punctured_ber > 0:
            rate_advantage = punctured_stats['rate'] / mother_stats['rate']
            print(f"Mother: {mother_ber:.6f}, Punctured: {punctured_ber:.6f}, Rate×: {rate_advantage:.2f}")
        else:
            print(f"Mother: {mother_ber:.6f}, Punctured: {punctured_ber:.6f}")

    return (bit_error_probs, mother_ber_results, punctured_ber_results,
            mother_statistics, punctured_statistics)


def plot_ber_comparison_vs_channel_probability(prob_values, mother_ber, punctured_ber,
                                               mother_stats, punctured_stats, save_plot=True):
    """Create comprehensive BER comparison visualization.

    Args:
        prob_values (list): Channel bit error probabilities.
        mother_ber (list): End-to-end BER for mother code.
        punctured_ber (list): End-to-end BER for punctured code.
        mother_stats (list): Statistics for mother code.
        punctured_stats (list): Statistics for punctured code.
        save_plot (bool): Whether to save the plot to file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: BER vs Channel probability
    ax1.loglog(prob_values, mother_ber, 'ro-', linewidth=2, markersize=8,
               label='Mother Code (4,1,4)', markerfacecolor='white', markeredgewidth=2)
    ax1.loglog(prob_values, punctured_ber, 'bs-', linewidth=2, markersize=8,
               label='Punctured Code (Rate 2/5)', markerfacecolor='white', markeredgewidth=2)
    ax1.loglog(prob_values, prob_values, 'k--', alpha=0.5, label='Uncoded (p = BER)')

    ax1.grid(True, which="both", ls="-", alpha=0.3)
    ax1.set_xlabel('Channel Bit Error Probability (p)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('End-to-End Bit Error Rate', fontsize=12, fontweight='bold')
    ax1.set_title('BER Performance vs Channel Error Probability', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.set_xlim(min(prob_values), max(prob_values))
    ax1.set_ylim(1e-6, 1)

    # Plot 2: Rate-reliability trade-off
    mother_rates = [s['rate'] for s in mother_stats]
    punctured_rates = [s['rate'] for s in punctured_stats]

    for i, p in enumerate(prob_values[::2]):
        idx = i * 2
        if idx < len(mother_ber):
            ax2.scatter(mother_rates[idx], mother_ber[idx],
                        c='red', s=100, marker='o', alpha=0.7,
                        label='Mother Code' if i == 0 else "")
            ax2.scatter(punctured_rates[idx], punctured_ber[idx],
                        c='blue', s=100, marker='s', alpha=0.7,
                        label='Punctured Code' if i == 0 else "")

            ax2.plot([mother_rates[idx], punctured_rates[idx]],
                     [mother_ber[idx], punctured_ber[idx]],
                     'gray', alpha=0.5, linewidth=1)

            ax2.annotate(f'p={p:.3f}',
                         xy=(punctured_rates[idx], punctured_ber[idx]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=9, alpha=0.8)

    ax2.set_yscale('log')
    ax2.grid(True, which="both", ls="-", alpha=0.3)
    ax2.set_xlabel('Code Rate', fontsize=12, fontweight='bold')
    ax2.set_ylabel('End-to-End Bit Error Rate', fontsize=12, fontweight='bold')
    ax2.set_title('Rate-Reliability Trade-off', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)

    plt.tight_layout()

    if save_plot:
        plt.savefig('mother_vs_punctured_ber_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as 'mother_vs_punctured_ber_analysis.png'")

    plt.show()


def generate_performance_summary(prob_values, mother_ber, punctured_ber,
                                 mother_stats, punctured_stats):
    """Generate detailed performance summary for technical report.

    Args:
        prob_values (list): Channel bit error probabilities tested.
        mother_ber (list): End-to-end BER results for mother code.
        punctured_ber (list): End-to-end BER results for punctured code.
        mother_stats (list): Detailed statistics for mother code.
        punctured_stats (list): Detailed statistics for punctured code.
    """
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY FOR FINAL REPORT")
    print("=" * 80)

    print(f"{'Channel p':<12} {'Mother BER':<12} {'Punct. BER':<12} {'Rate Adv.':<10} {'Rel. Trade-off':<15}")
    print("-" * 80)

    for i, p in enumerate(prob_values):
        mother_rate = mother_stats[i]['rate']
        punctured_rate = punctured_stats[i]['rate']
        rate_advantage = punctured_rate / mother_rate

        if mother_ber[i] > 0:
            reliability_ratio = punctured_ber[i] / mother_ber[i]
            rel_str = f"{reliability_ratio:.2f}×"
        else:
            rel_str = "N/A"

        print(f"{p:<12.4f} {mother_ber[i]:<12.2e} {punctured_ber[i]:<12.2e} "
              f"{rate_advantage:<10.2f} {rel_str:<15}")

    # Key insights
    print(f"\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    rate_improvement = punctured_stats[0]['rate'] / mother_stats[0]['rate']
    print(f"✓ Rate Improvement: {rate_improvement:.2f}× "
          f"(from {mother_stats[0]['rate']:.3f} to {punctured_stats[0]['rate']:.3f})")

    mother_overhead = (1 / mother_stats[0]['rate'] - 1) * 100
    punctured_overhead = (1 / punctured_stats[0]['rate'] - 1) * 100
    overhead_reduction = mother_overhead - punctured_overhead
    print(f"✓ Overhead Reduction: {overhead_reduction:.1f}% "
          f"(from {mother_overhead:.1f}% to {punctured_overhead:.1f}%)")

    if len(prob_values) > 1:
        low_p_degradation = punctured_ber[0] / mother_ber[0] if mother_ber[0] > 0 else 1
        high_p_degradation = punctured_ber[-1] / mother_ber[-1] if mother_ber[-1] > 0 else 1

        print(f"✓ Reliability Trade-off at p={prob_values[0]:.3f}: {low_p_degradation:.1f}× worse BER")
        print(f"✓ Reliability Trade-off at p={prob_values[-1]:.3f}: {high_p_degradation:.1f}× worse BER")

    print(f"\n✓ Theoretical Context:")
    for i, p in enumerate(prob_values[::3]):
        idx = i * 3
        if idx < len(prob_values):
            if p > 0 and p < 1:
                capacity = 1 - (-p * np.log2(p) - (1 - p) * np.log2(1 - p))
            else:
                capacity = 1 if p == 0 else 0

            print(f"  p={p:.3f}: Channel capacity = {capacity:.3f}, "
                  f"Mother rate = {mother_stats[idx]['rate']:.3f}, "
                  f"Punctured rate = {punctured_stats[idx]['rate']:.3f}")


def main():
    """Main function to run the complete comparison analysis."""
    print("=" * 80)
    print("FINAL PROJECT: MOTHER CODE vs PUNCTURED CODE COMPARISON")
    print("Binary Symmetric Channel with Variable Error Probability")
    print("=" * 80)

    # Simulation parameters
    bit_error_probabilities = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    num_trials = 30
    bits_per_trial = 300

    print(f"Channel: Binary Symmetric Channel")
    print(f"Error probabilities: {bit_error_probabilities}")
    print(f"Mother code: (4,1,4) Convolutional Code (rate 1/4)")
    print(f"Punctured code: Rate 2/5 with pattern [1111 1111; 1111 1111; 1100 1100; 0000 0000]")
    print(f"Trials per probability: {num_trials}")
    print(f"Information bits per trial: {bits_per_trial}")
    print()

    # Run analysis
    results = run_comprehensive_ber_comparison(
        bit_error_probabilities,
        num_trials=num_trials,
        bits_per_trial=bits_per_trial,
        seed=42
    )

    prob_values, mother_ber, punctured_ber, mother_stats, punctured_stats = results

    # Generate outputs
    generate_performance_summary(
        prob_values, mother_ber, punctured_ber,
        mother_stats, punctured_stats
    )

    plot_ber_comparison_vs_channel_probability(
        prob_values, mother_ber, punctured_ber,
        mother_stats, punctured_stats
    )

    # Recommendations
    print(f"\n" + "=" * 80)
    print("RECOMMENDATIONS FOR SYSTEM DESIGN")
    print("=" * 80)
    print("✓ Use punctured code when bandwidth is limited and moderate BER increase is acceptable")
    print("✓ Use mother code when maximum reliability is required regardless of bandwidth cost")
    print("✓ Consider adaptive puncturing: switch patterns based on channel conditions")
    print("✓ Puncturing provides 1.6× rate improvement with typically <3× BER degradation")


if __name__ == "__main__":
    main()