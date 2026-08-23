import pytest
import torch

from ts_trans.model.embedding import ScalarEmbedding


def controlled_input():
    return torch.tensor(
        [
            [[0.0], [1.0], [2.0], [3.0], [4.0]],
            [[-1.0], [-0.5], [0.5], [1.5], [2.5]],
        ],
        dtype=torch.float32,
    )


def test_scalar_embedding_shapes_parameters_and_gradients():
    torch.manual_seed(20260823)
    model = ScalarEmbedding(d_model=8)
    output = model(controlled_input())

    assert output.shape == (2, 5, 8)
    expected_shapes = {
        "linear1.weight": (8, 1),
        "linear1.bias": (8,),
        "linear2.weight": (8, 8),
        "linear2.bias": (8,),
    }
    assert {name: tuple(parameter.shape) for name, parameter in model.named_parameters()} == expected_shapes
    assert sum(parameter.numel() for parameter in model.parameters()) == 88

    output.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_different_scalar_inputs_produce_different_embeddings():
    torch.manual_seed(20260823)
    model = ScalarEmbedding(d_model=8)
    output = model(controlled_input())

    assert not torch.equal(output[0, 0], output[0, 1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_scalar_embedding_cuda_forward_and_gradients():
    torch.manual_seed(20260823)
    model = ScalarEmbedding(d_model=8).cuda()
    output = model(controlled_input().cuda())

    assert output.shape == (2, 5, 8)
    assert output.is_cuda
    output.sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
