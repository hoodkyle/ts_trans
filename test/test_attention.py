import pytest
import torch

from ts_trans.model.attention import SelfAttention


def controlled_input():
    return torch.arange(32, dtype=torch.float32).reshape(1, 4, 8) / 10.0


def test_attention_parameters_and_output_shapes():
    torch.manual_seed(20260823)
    model = SelfAttention(d_model=8)
    output, weights = model(controlled_input())

    expected_shapes = {
        "W_q.weight": (8, 8),
        "W_k.weight": (8, 8),
        "W_v.weight": (8, 8),
    }
    assert {name: tuple(parameter.shape) for name, parameter in model.named_parameters()} == expected_shapes
    assert sum(parameter.numel() for parameter in model.parameters()) == 3 * 8 * 8
    assert output.shape == (1, 4, 8)
    assert weights.shape == (1, 4, 4)


def test_attention_weights_are_normalized_and_causal():
    torch.manual_seed(20260823)
    _, weights = SelfAttention(d_model=8)(controlled_input())

    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1, 4))
    assert bool((weights >= 0).all())
    for row in range(4):
        torch.testing.assert_close(weights[0, row, row + 1 :], torch.zeros(3 - row))
    torch.testing.assert_close(weights[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_attention_is_independent_of_future_inputs():
    torch.manual_seed(20260823)
    model = SelfAttention(d_model=8)
    first = controlled_input()
    second = first.clone()
    second[:, 2:, :] += 100.0

    first_output, _ = model(first)
    second_output, _ = model(second)
    torch.testing.assert_close(first_output[:, :2], second_output[:, :2])


def test_attention_gradients_are_finite():
    torch.manual_seed(20260823)
    model = SelfAttention(d_model=8)
    output, _ = model(controlled_input())
    output.square().mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.shape == parameter.shape
        assert bool(torch.isfinite(parameter.grad).all()), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_attention_cuda_matches_cpu_and_has_finite_gradients():
    torch.manual_seed(20260823)
    cpu_model = SelfAttention(d_model=8)
    cuda_model = SelfAttention(d_model=8).cuda()
    cuda_model.load_state_dict(cpu_model.state_dict())
    cpu_input = controlled_input()

    cpu_output, cpu_weights = cpu_model(cpu_input)
    cuda_output, cuda_weights = cuda_model(cpu_input.cuda())
    torch.testing.assert_close(cuda_output.cpu(), cpu_output)
    torch.testing.assert_close(cuda_weights.cpu(), cpu_weights)
    assert cuda_output.device.type == "cuda"
    assert cuda_weights.device.type == "cuda"

    cuda_output.square().mean().backward()
    for parameter in cuda_model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
