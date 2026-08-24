import pytest
import torch

from ts_trans.model.feedforward import FeedForward


def controlled_input():
    return torch.arange(80, dtype=torch.float32).reshape(2, 5, 8) / 10.0


def test_feedforward_parameters_and_shapes():
    torch.manual_seed(20260824)
    model = FeedForward(d_model=8, d_ff=16)
    output = model(controlled_input())

    expected_shapes = {
        "linear1.weight": (16, 8),
        "linear1.bias": (16,),
        "linear2.weight": (8, 16),
        "linear2.bias": (8,),
    }
    assert {name: tuple(parameter.shape) for name, parameter in model.named_parameters()} == expected_shapes
    assert sum(parameter.numel() for parameter in model.parameters()) == 16 * 8 + 16 + 8 * 16 + 8
    assert output.shape == (2, 5, 8)


def test_feedforward_is_tokenwise():
    torch.manual_seed(20260824)
    model = FeedForward(d_model=8, d_ff=16)
    first = controlled_input()
    second = first.clone()
    second[0, 2] += 100.0

    first_output = model(first)
    second_output = model(second)
    torch.testing.assert_close(first_output[0, :2], second_output[0, :2])
    torch.testing.assert_close(first_output[0, 3:], second_output[0, 3:])
    assert not torch.equal(first_output[0, 2], second_output[0, 2])


def test_feedforward_gradients_are_finite():
    torch.manual_seed(20260824)
    model = FeedForward(d_model=8, d_ff=16)
    model(controlled_input()).square().mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.shape == parameter.shape
        assert bool(torch.isfinite(parameter.grad).all()), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_feedforward_cuda_matches_cpu_and_has_finite_gradients():
    torch.manual_seed(20260824)
    cpu_model = FeedForward(d_model=8, d_ff=16)
    cuda_model = FeedForward(d_model=8, d_ff=16).cuda()
    cuda_model.load_state_dict(cpu_model.state_dict())
    cpu_input = controlled_input()

    cpu_output = cpu_model(cpu_input)
    cuda_output = cuda_model(cpu_input.cuda())
    assert cuda_output.shape == (2, 5, 8)
    assert cuda_output.device.type == "cuda"
    torch.testing.assert_close(cuda_output.cpu(), cpu_output)

    cuda_output.square().mean().backward()
    for parameter in cuda_model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
