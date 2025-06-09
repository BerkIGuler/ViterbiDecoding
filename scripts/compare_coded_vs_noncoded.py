import numpy as np
import matplotlib.pyplot as plt

from src.modulator import PAMModulator
from src.channels import BSC
from src.input_source import RandomBinarySource
from src.conv_encoder import ConvolutionalEncoder
from src.viterbi_decoder import ViterbiDecoder


def simulate_noncoded_transmission(data_bits, snr_db, amplitude=1.0):
    """
    Simulate non-coded transmission with basic thresholding.

    Returns:
    tuple: (transmitted_bits, received_bits, ber)
    """
    modulator = PAMModulator(amplitude=amplitude)
    symbols = modulator.modulate(data_bits)

    channel = BSC(snr_db=snr_db, amplitude=amplitude)
    _, received_bits = channel.transmit(symbols)

    errors = np.sum(np.array(data_bits) != np.array(received_bits))
    ber = errors / len(data_bits)

    return data_bits, received_bits, ber


def simulate_coded_transmission(data_bits, snr_db, amplitude=1.0):
    """
    Simulate coded transmission with convolutional encoding and Viterbi decoding.

    Returns:
    tuple: (transmitted_bits, decoded_bits, ber)
    """
    block_length = len(data_bits)

    if isinstance(data_bits, np.ndarray):
        data_bits = data_bits.tolist()

    encoder = ConvolutionalEncoder(block_length)
    encoded_bits = encoder.encode(data_bits)

    modulator = PAMModulator(amplitude=amplitude)
    symbols = modulator.modulate(encoded_bits)

    channel = BSC(snr_db=snr_db, amplitude=amplitude)
    _, received_coded_bits = channel.transmit(symbols)

    decoder = ViterbiDecoder(block_length)
    decoded_bits = decoder.decode(received_coded_bits)

    errors = np.sum(np.array(data_bits) != np.array(decoded_bits))
    ber = errors / len(data_bits)

    return data_bits, decoded_bits, ber


def run_ber_comparison(snr_values, num_trials=10, bits_per_trial=1000, amplitude=1.0, seed=42):
    """
    Run BER comparison between coded and non-coded systems.

    Parameters:
    snr_values: List of SNR values in dB
    num_trials: Number of Monte Carlo trials per SNR
    bits_per_trial: Number of information bits per trial
    amplitude: Signal amplitude
    seed: Random seed for reproducibility

    Returns:
    tuple: (snr_values, noncoded_ber, coded_ber)
    """
    np.random.seed(seed)
    source = RandomBinarySource(p_1=0.5, seed=seed)

    noncoded_ber_results = []
    coded_ber_results = []

    print("=== BER Comparison: Coded vs Non-Coded ===")
    print(f"Trials per SNR: {num_trials}")
    print(f"Bits per trial: {bits_per_trial}")
    print(f"Total bits per SNR: {num_trials * bits_per_trial}")
    print("-" * 60)

    for snr_db in snr_values:
        print(f"Testing SNR = {snr_db:2d} dB...", end=" ")

        noncoded_errors = 0
        coded_errors = 0
        total_bits = 0

        for trial in range(num_trials):
            data_bits = source.generate(bits_per_trial)
            total_bits += len(data_bits)

            _, received_noncoded, _ = simulate_noncoded_transmission(
                data_bits, snr_db, amplitude
            )
            noncoded_errors += np.sum(np.array(data_bits) != np.array(received_noncoded))

            _, decoded_bits, _ = simulate_coded_transmission(
                data_bits, snr_db, amplitude
            )
            coded_errors += np.sum(np.array(data_bits) != np.array(decoded_bits))

        noncoded_ber = noncoded_errors / total_bits
        coded_ber = coded_errors / total_bits

        noncoded_ber_results.append(noncoded_ber)
        coded_ber_results.append(coded_ber)

        if noncoded_ber > 0 and coded_ber > 0:
            coding_gain = 10 * np.log10(noncoded_ber / coded_ber)
            print(f"Non-coded: {noncoded_ber:.6f}, Coded: {coded_ber:.6f}, Gain: {coding_gain:.1f} dB")
        elif coded_ber == 0:
            print(f"Non-coded: {noncoded_ber:.6f}, Coded: {coded_ber:.6f}, Gain: >20 dB")
        else:
            print(f"Non-coded: {noncoded_ber:.6f}, Coded: {coded_ber:.6f}")

    return snr_values, noncoded_ber_results, coded_ber_results


def plot_ber_comparison(snr_values, noncoded_ber, coded_ber, save_plot=True):
    """
    Create BER comparison plot.
    """
    plt.figure(figsize=(10, 6))

    plt.semilogy(snr_values, noncoded_ber, 'ro-', linewidth=2, markersize=8,
                 label='Non-Coded (Basic Thresholding)', markerfacecolor='white', markeredgewidth=2)
    plt.semilogy(snr_values, coded_ber, 'bs-', linewidth=2, markersize=8,
                 label='Coded (Convolutional + Viterbi)', markerfacecolor='white', markeredgewidth=2)

    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.xlabel('Signal-to-Noise Ratio (SNR) [dB]', fontsize=12, fontweight='bold')
    plt.ylabel('Bit Error Rate (BER)', fontsize=12, fontweight='bold')
    plt.title('BER Performance: Coded vs Non-Coded Transmission\n(4,1,4) Convolutional Code with Viterbi Decoding',
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')

    plt.xlim(min(snr_values) - 0.5, max(snr_values) + 0.5)
    plt.ylim(1e-5, 1)

    plt.tight_layout()

    if save_plot:
        plt.savefig('ber_comparison_coded_vs_noncoded.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as 'ber_comparison_coded_vs_noncoded.png'")

    plt.show()


def main():
    """
    Main function to run the complete BER comparison.
    """
    # Simulation parameters
    snr_values = np.arange(-6, 10, 2)
    num_trials = 25
    bits_per_trial = 10000
    amplitude = 1.0

    print(f"SNR range: {min(snr_values)} to {max(snr_values)} dB")
    print(f"Code: (4,1,4) Convolutional Code")
    print(f"Modulation: Binary PAM (0→-{amplitude}, 1→+{amplitude})")
    print(f"Channel: Binary Symmetric Channel with AWGN")
    print(f"Decoding: Viterbi Algorithm (Hard Decision)")
    print()

    snr_db, noncoded_ber, coded_ber = run_ber_comparison(
        snr_values=snr_values,
        num_trials=num_trials,
        bits_per_trial=bits_per_trial,
        amplitude=amplitude,
        seed=42
    )

    print("\n" + "=" * 60)
    print("SUMMARY RESULTS")
    print("=" * 60)
    print(f"{'SNR [dB]':<8} {'Non-Coded BER':<15} {'Coded BER':<15} {'Coding Gain [dB]':<15}")
    print("-" * 60)

    for snr, nc_ber, c_ber in zip(snr_db, noncoded_ber, coded_ber):
        if nc_ber > 0 and c_ber > 0:
            gain = 10 * np.log10(nc_ber / c_ber)
            gain_str = f"{gain:.1f}"
        elif c_ber == 0:
            gain_str = ">20.0"
        else:
            gain_str = "N/A"

        print(f"{snr:<8} {nc_ber:<15.6f} {c_ber:<15.6f} {gain_str:<15}")

    plot_ber_comparison(snr_db, noncoded_ber, coded_ber)

    print(f"\n" + "=" * 60)
    print("PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Find SNR for specific BER targets
    target_bers = [1e-2, 1e-3, 1e-4]

    for target_ber in target_bers:
        # Find required SNR for non-coded
        nc_snr = np.interp(target_ber, noncoded_ber[::-1], snr_db[::-1])
        c_snr = np.interp(target_ber, coded_ber[::-1], snr_db[::-1])

        if not np.isnan(nc_snr) and not np.isnan(c_snr):
            snr_gain = nc_snr - c_snr
            print(f"For BER = {target_ber:.0e}:")
            print(f"  Non-coded requires: {nc_snr:.1f} dB")
            print(f"  Coded requires:     {c_snr:.1f} dB")
            print(f"  SNR gain:          {snr_gain:.1f} dB")
            print()


if __name__ == "__main__":
    main()