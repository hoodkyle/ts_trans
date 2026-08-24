import pytest
import torch

from ts_trans.model.block import TransformerBlock


def controlled_input():
    return torch.arange(80, dtype=torch.float32).reshape(2, 5, 8) / 10.0


def test_block_shapes_and_parameter_count():
    torch.manual_seed(20260824)
    block = TransformerBlock(d_model=8, d_ff=16)
    output, weights = block(controlled_input())

    expected_shapes = {
        "norm1.weight": (8,),
        "norm1.bias": (8,),
        "attention.W_q.weight": (8, 8),
        "attention.W_k.weight": (8, 8),
        "attention.W_v.weight": (8, 8),
        "norm2.weight": (8,),
        "norm2.bias": (8,),
        "feedforward.linear1.weight": (16, 8),
        "feedforward.linear1.bias": (16,),
        "feedforward.linear2.weight": (8, 16),
        "feedforward.linear2.bias": (8,),
    }
    assert {name: tuple(parameter.shape) for name, parameter in block.named_parameters()} == expected_shapes
    assert sum(parameter.numel() for parameter in block.parameters()) == 16 + 192 + 16 + 280
    assert output.shape == (2, 5, 8)
    assert weights.shape == (2, 5, 5)


def test_block_matches_manual_residual_reconstruction():
    torch.manual_seed(20260824)
    block = TransformerBlock(d_model=8, d_ff=16)
    x = controlled_input()

    block_output, block_weights = block(x)
    normed1 = block.norm1(x)
    attn_output, manual_weights = block.attention(normed1)
    after_attn = x + attn_output
    normed2 = block.norm2(after_attn)
    ff_output = block.feedforward(normed2)
    manual_output = after_attn + ff_output

    torch.testing.assert_close(manual_output, block_output)
    torch.testing.assert_close(manual_weights, block_weights)


def test_block_preserves_causal_independence():
    torch.manual_seed(20260824)
    block = TransformerBlock(d_model=8, d_ff=16)
    first = controlled_input()
    second = first.clone()
    second[:, 3:, :] += 100.0

    first_output, _ = block(first)
    second_output, _ = block(second)
    torch.testing.assert_close(first_output[:, :3], second_output[:, :3])


def test_block_attention_mask_and_layer_norm_properties():
    torch.manual_seed(20260824)
    block = TransformerBlock(d_model=8, d_ff=16)
    x = controlled_input()
    _, weights = block(x)

    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2, 5))
    assert bool((weights >= 0).all())
    assert bool(torch.all(weights.triu(diagonal=1) == 0))
    torch.testing.assert_close(weights[:, 0], torch.tensor([[1.0, 0, 0, 0, 0]]).expand(2, -1))

    normalized = block.norm1(x)
    means = normalized.mean(dim=-1)
    variances = normalized.var(dim=-1, unbiased=False)
    torch.testing.assert_close(means, torch.zeros_like(means), atol=5e-6, rtol=0)
    torch.testing.assert_close(variances, torch.ones_like(variances), atol=3e-4, rtol=0)


def test_block_gradients_are_finite():
    torch.manual_seed(20260824)
    block = TransformerBlock(d_model=8, d_ff=16)
    block(controlled_input())[0].square().mean().backward()

    for name, parameter in block.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.shape == parameter.shape
        assert bool(torch.isfinite(parameter.grad).all()), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_block_cuda_matches_cpu_and_has_finite_gradients():
    torch.manual_seed(20260824)
    cpu_block = TransformerBlock(d_model=8, d_ff=16)
    cuda_block = TransformerBlock(d_model=8, d_ff=16).cuda()
    cuda_block.load_state_dict(cpu_block.state_dict())
    cpu_input = controlled_input()

    cpu_output, cpu_weights = cpu_block(cpu_input)
    cuda_output, cuda_weights = cuda_block(cpu_input.cuda())
    assert cuda_output.shape == (2, 5, 8)
    assert cuda_output.device.type == "cuda"
    assert cuda_weights.device.type == "cuda"
    torch.testing.assert_close(cuda_output.cpu(), cpu_output)
    torch.testing.assert_close(cuda_weights.cpu(), cpu_weights)

    cuda_output.square().mean().backward()
    for parameter in cuda_block.parameters():
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())
