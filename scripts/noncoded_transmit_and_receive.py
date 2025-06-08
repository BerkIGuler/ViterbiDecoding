import numpy as np

from src.modulator import PAMModulator
from src.channels import BSC
from src.input_source import RandomBinarySource


def main():
    # Set seed for reproducible results
    np.random.seed(42)

    # Parameters
    block_length = 16
    amplitude = 1.0
    snr_db = 10.0

    print("=== Communication System Test ===")

    # 1. Generate random data
    source = RandomBinarySource(p_1=0.5, seed=42)
    data_bits = source.generate(block_length)
    print(f"Random data bits: {data_bits}")
    print(f"Source probabilities: P(0)={source.p_0}, P(1)={source.p_1}")

    # 2. PAM Modulation
    modulator = PAMModulator(amplitude=amplitude)
    symbols = modulator.modulate(data_bits)
    print(f"Modulated symbols: {symbols}")
    print(f"Signal power: {modulator.get_signal_power()}")

    # 3. BSC Channel
    channel = BSC(snr_db=snr_db, amplitude=amplitude)
    channel_params = channel.get_channel_parameters()
    print(f"\nChannel parameters:")
    for key, value in channel_params.items():
        print(f"  {key}: {value:.6f}")

    # 4. Transmission
    noisy_symbols, received_bits = channel.transmit(symbols)
    print(f"\nNoisy symbols: {noisy_symbols}")
    print(f"Received bits: {received_bits}")

    # 5. Calculate bit error rate
    errors = np.sum(np.array(data_bits) != np.array(received_bits))
    ber = errors / len(data_bits)
    print(f"\nTransmission Results:")
    print(f"  Transmitted: {data_bits}")
    print(f"  Received:    {list(received_bits)}")
    print(f"  Errors: {errors}/{len(data_bits)}")
    print(f"  BER: {ber:.4f}")

    # 6. Test with different SNR values
    print(f"\n=== BER vs SNR Test ===")
    test_length = 1000
    snr_values = [0, 5, 10, 15, 20]

    for snr in snr_values:
        # Generate test data
        test_data = source.generate(test_length)

        # Modulate and transmit
        test_symbols = modulator.modulate(test_data)
        test_channel = BSC(snr_db=snr, amplitude=amplitude)
        _, test_received = test_channel.transmit(test_symbols)

        # Calculate BER
        test_errors = np.sum(np.array(test_data) != np.array(test_received))
        test_ber = test_errors / test_length

        print(f"SNR = {snr:2d} dB: BER = {test_ber:.6f} ({test_errors}/{test_length} errors)")


if __name__ == "__main__":
    main()