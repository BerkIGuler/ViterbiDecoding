# Rate-Compatible Punctured Convolutional Code BER Analysis

This project implements and compares the performance of a rate 1/4 convolutional mother code against its punctured variant through comprehensive Bit Error Rate (BER) analysis over Binary Symmetric Channels (BSC).

## Project Overview

The system simulates two convolutional coding schemes:

1. **Mother Code (4,1,4)**: A rate 1/4 convolutional code with constraint length 4
2. **Punctured Code**: A rate 2/5 code derived from the mother code using puncturing

Both codes are evaluated under identical channel conditions to compare their error correction capabilities.

## Code Structure

### Generator Polynomials

The mother code uses four generator sequences:
- g₁ = (1 0 0 1 1)
- g₂ = (1 1 1 0 1) 
- g₃ = (1 0 1 1 1)
- g₄ = (1 1 0 1 1)

### Puncturing Matrix

The puncturing matrix (P = 8) removes specific bits from the mother code output:

```
1111 1111
1111 1111  
1100 1100
0000 0000
```

Where `1` indicates transmitted bits and `0` indicates punctured (deleted) bits.

## How the Script Works

### Simulation Flow

```
Information Bits → Convolutional Encoder → [Puncturing] → BSC Channel → Viterbi Decoder → Decoded Bits
```
The script performs the following analysis:
- Tests multiple channel error probabilities
- Runs Monte Carlo trials for statistical significance
- Compares mother code vs punctured code performance
- Generates comparative BER plots

## Installation and Dependencies
Ensure you have the required Python packages:

```bash
pip install numpy matplotlib tqdm
```

## Usage

### Basic Execution

Run the complete BER analysis:

```bash
python ber_comparison_punctured_vs_nonpunctured.py
```

### Output
The script generates:
- Console progress updates via tqdm
- BER performance plot (`ber_vs_channel_probability.png`)
- Comparative analysis showing:
  - Mother code performance (red circles)
  - Punctured code performance (blue squares)
  - Uncoded baseline (black dashed line)


## Customization

### Modifying Simulation Parameters

```python
# Increase statistical accuracy
num_trials = 1000
bits_per_trial = 50000

# Test different channel conditions  
bit_error_probabilities = np.logspace(-3, -1, 20)

# Change random seed for different data patterns
seed = 123
```

### Alternative Puncturing Patterns
Modify the puncturing matrix in `src/puncturing.py` to test different rate-compatible codes.

## Theoretical Background
This implementation demonstrates:
- **Rate-Compatible Punctured Convolutional (RCPC) Codes**: Systematic method for creating multiple code rates from a single mother code
- **Viterbi Decoding**: Maximum likelihood sequence estimation for convolutional codes
- **Performance Trade-offs**: Rate vs. error correction capability in digital communications

## Troubleshooting
**Common Issues:**
- Check memory usage for large simulation parameters

For detailed algorithm implementations, refer to the individual modules in the `src/` directory.
