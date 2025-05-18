import torch
import os
from project.models.miniGRU import MinimalRNN


def test_parallelScan_randomFloats():
    # Test the parallel_scan_log function with random floats
    batch_size = 10000
    seq_len = 200
    input_size = 2

    # coefs = torch.ones(batch_size, seq_len, input_size) * 1
    # values = torch.arange(batch_size * (seq_len + 1) *
    #                       input_size).reshape(batch_size, seq_len + 1, input_size).float()
    # log_coeffs = torch.log(coefs)
    # log_values = torch.log(values)

    log_coeffs = torch.randn(batch_size, seq_len, input_size)
    log_values = torch.randn(batch_size, seq_len + 1, input_size)

    # Call the parallel_scan_log function
    result = MinimalRNN.parallel_scan_log(log_coeffs, log_values)

    # Expected result
    coeffs = log_coeffs.exp()
    # Pad the coeffs tensor with a leading zero in sequence dimension
    coeffs = torch.cat(
        (torch.zeros(batch_size, 1, input_size), coeffs), dim=1)
    values = log_values.exp()
    print("coeffs", coeffs)
    print("values", values)
    expected_result = torch.zeros(batch_size, seq_len + 2, input_size)
    for i in range(batch_size):
        for j in range(seq_len+1):
            expected_result[i, j + 1] = coeffs[i, j] * \
                expected_result[i, j] + values[i, j]
    expected_result = expected_result[:, 2:]
    print("expected_result", expected_result)
    print("result", result)
    # Check if the result is close to the expected result
    assert torch.allclose(result, expected_result,
                          atol=1e-6), "Output mismatch"

    # Check the shape of the result
    assert result.shape == (batch_size, seq_len,
                            input_size), "Output shape mismatch"
