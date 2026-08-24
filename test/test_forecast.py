import pytest
import torch

from ts_trans.model.block import TransformerBlock
from ts_trans.model.forecast import TimeSeriesTransformer


def controlled_input():
    return torch.arange(10, dtype=torch.float32).reshape(2, 5, 1) / 10.0


def test_model_hierarchy_and_parameter_count():
    torch.manual_seed(20260824)
    model = TimeSeriesTransformer(d_model=8, d_ff=16, n_blocks=2)

    assert isinstance(model.embedding, torch.nn.Module)
    assert isinstance(model.position, torch.nn.Module)
    assert len(model.blocks) == 2
    assert all(isinstance(block, TransformerBlock) for block in model.blocks)
    assert isinstance(model.head, torch.nn.Linear)
    assert sum(parameter.numel() for parameter in model.embedding.parameters()) == 88
    assert sum(parameter.numel() for parameter in model.blocks[0].parameters()) == 504
    assert sum(parameter.numel() for parameter in model.blocks[1].parameters()) == 504
    assert sum(parameter.numel() for parameter in model.head.parameters()) == 9
    assert sum(parameter.numel() for parameter in model.parameters()) == 1105
    assert sum(parameter.numel() for parameter in model.position.parameters()) == 0

    for first, second in zip(model.blocks[0].parameters(), model.blocks[1].parameters()):
        assert first.data_ptr() != second.data_ptr()


def test_full_forward_shapes_and_manual_reconstruction():
    torch.manual_seed(20260824)
    model = TimeSeriesTransformer(d_model=8, d_ff=16, n_blocks=2)
    x = controlled_input()

    output = model(x)
    embedded = model.embedding(x)
    positioned = model.position(embedded)
    z = positioned
    intermediate_shapes = [tuple(z.shape)]
    for block in model.blocks:
        z, _ = block(z)
        intermediate_shapes.append(tuple(z.shape))
    last = z[:, -1, :]
    manual_forecast = model.head(last)

    assert output.shape == (2, 1)
    assert intermediate_shapes == [(2, 5, 8), (2, 5, 8), (2, 5, 8)]
    assert last.shape == (2, 8)
    torch.testing.assert_close(manual_forecast, output)
    torch.testing.assert_close(model.head(z[:, -1, :]), output)


def test_causal_prefixes_are_independent_of_future_inputs():
    torch.manual_seed(20260824)
    model = TimeSeriesTransformer(d_model=8, d_ff=16, n_blocks=2)
    first = controlled_input()
    second = first.clone()
    second[:, 3:, :] += 100.0

    first_z = model.position(model.embedding(first))
    second_z = model.position(model.embedding(second))
    for block in model.blocks:
        first_z, _ = block(first_z)
        second_z, _ = block(second_z)
        torch.testing.assert_close(first_z[:, :3], second_z[:, :3])


def test_end_to_end_gradients_are_finite():
    torch.manual_seed(20260824)
    model = TimeSeriesTransformer(d_model=8, d_ff=16, n_blocks=2)
    model(controlled_input()).square().mean().backward()

    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.shape == parameter.shape
        assert bool(torch.isfinite(parameter.grad).all()), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_full_model_cuda_matches_cpu_and_has_finite_gradients():
    torch.manual_seed(20260824)
    cpu_model = TimeSeriesTransformer(d_model=8, d_ff=16, n_blocks=2)
    cuda_model = TimeSeriesTransformer(d_model=8, d_ff=16, n_blocks=2).cuda()
    cuda_model.load_state_dict(cpu_model.state_dict())
    cpu_input = controlled_input()

    cpu_output = cpu_model(cpu_input)
    cuda_output = cuda_model(cpu_input.cuda())
    assert cuda_output.shape == (2, 1)
    assert cuda_output.device.type == "cuda"
    assert cuda_model.position.pe.device.type == "cuda"
    torch.testing.assert_close(cuda_output.cpu(), cpu_output)

    cuda_output.square().mean().backward()
    for parameter in cuda_model.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
