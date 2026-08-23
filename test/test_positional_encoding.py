import math

import pytest
import torch

from ts_trans.model.embedding import PositionalEncoding, ScalarEmbedding


def test_positional_matrix_and_registration():
    encoding = PositionalEncoding(d_model=8, max_len=10)

    assert encoding.pe.shape == (10, 8)
    assert "pe" in dict(encoding.named_buffers())
    assert "pe" not in dict(encoding.named_parameters())
    assert sum(parameter.numel() for parameter in encoding.parameters()) == 0

    expected_row_zero = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    torch.testing.assert_close(encoding.pe[0], expected_row_zero)


def test_forward_adds_positions_and_broadcasts_across_batch():
    encoding = PositionalEncoding(d_model=8, max_len=10)
    zeros = torch.zeros(2, 5, 8)
    zero_output = encoding(zeros)

    assert zero_output.shape == (2, 5, 8)
    torch.testing.assert_close(zero_output[0], encoding.pe[:5])
    torch.testing.assert_close(zero_output[0], zero_output[1])

    nonzero = torch.arange(80, dtype=torch.float32).reshape(2, 5, 8)
    torch.testing.assert_close(encoding(nonzero), nonzero + encoding.pe[:5])


def test_first_frequency_factors_are_the_expected_values():
    factors = torch.exp(
        torch.arange(0, 8, 2, dtype=torch.float32) * (-math.log(10000.0) / 8)
    )
    expected = torch.tensor([10000.0 ** (-k / 8) for k in (0, 2, 4, 6)])
    torch.testing.assert_close(factors, expected)


def test_scalar_embedding_then_positional_encoding_shapes():
    raw = torch.arange(10, dtype=torch.float32).reshape(2, 5, 1)
    embedded = ScalarEmbedding(d_model=8)(raw)
    positioned = PositionalEncoding(d_model=8, max_len=10)(embedded)

    assert embedded.shape == (2, 5, 8)
    assert positioned.shape == (2, 5, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_positional_encoding_cuda_and_cpu_agree():
    torch.manual_seed(20260823)
    cpu_encoding = PositionalEncoding(d_model=8, max_len=10)
    cuda_encoding = PositionalEncoding(d_model=8, max_len=10).cuda()
    cuda_encoding.load_state_dict(cpu_encoding.state_dict())

    cpu_input = torch.arange(80, dtype=torch.float32).reshape(2, 5, 8)
    cuda_output = cuda_encoding(cpu_input.cuda())
    cpu_output = cpu_encoding(cpu_input)

    assert cuda_encoding.pe.device.type == "cuda"
    assert cuda_output.shape == (2, 5, 8)
    assert cuda_output.device.type == "cuda"
    torch.testing.assert_close(cuda_output.cpu(), cpu_output)
