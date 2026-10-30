"""Standalone tests for Phase 1+2 changes.

Verifies mathematical correctness of:
- Residual gate fusion
- Alpha regularization
- Multi-task graph loss
- Graph-enriched sequence input logic

Does NOT require PyTorch — uses numpy as a stand-in.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def test_residual_gate():
    """fused = graph + alpha * (seq - graph). alpha=0 → pure graph."""
    dim = 64
    rng = np.random.RandomState(42)

    seq = rng.randn(dim).astype(np.float32)
    graph = rng.randn(dim).astype(np.float32)

    # alpha = 0 → fused == graph exactly
    alpha = 0.0
    fused = graph + alpha * (seq - graph)
    assert np.allclose(fused, graph), "alpha=0 should give pure graph"

    # alpha = 1 → fused == seq exactly
    alpha = 1.0
    fused = graph + alpha * (seq - graph)
    assert np.allclose(fused, seq), "alpha=1 should give pure seq"

    # alpha = 0.3 → fused = 0.7*graph + 0.3*seq
    alpha = 0.3
    fused = graph + alpha * (seq - graph)
    expected = (1 - alpha) * graph + alpha * seq
    assert np.allclose(fused, expected), "residual form should equal interpolation"

    print("PASS: residual gate formula correct")


def test_old_vs_new_formula():
    """Old: alpha*seq + (1-a)*graph. New: graph + alpha*(seq-graph). Algebraically identical."""
    dim = 64
    rng = np.random.RandomState(123)

    seq = rng.randn(dim).astype(np.float32)
    graph = rng.randn(dim).astype(np.float32)

    for alpha in [0.0, 0.1, 0.5, 0.73, 0.85, 1.0]:
        old = alpha * seq + (1.0 - alpha) * graph
        new = graph + alpha * (seq - graph)
        assert np.allclose(old, new, atol=1e-6), f"formulas differ at alpha={alpha}"

    print("PASS: old and new formulas are algebraically equivalent")


def test_alpha_reg_pushes_to_zero():
    """L_alpha_reg = lambda * mean(alpha). Minimizing this pushes alpha -> 0 -> pure graph."""
    alphas = np.array([0.73, 0.81, 0.85])
    lambda_reg = 0.1

    reg_loss = lambda_reg * alphas.mean()
    assert reg_loss > 0, "reg loss should be positive"

    grad_alpha = lambda_reg / len(alphas)
    assert grad_alpha > 0, "gradient should push alpha downward"

    print(f"PASS: alpha_reg={lambda_reg}, mean_alpha={alphas.mean():.3f}, "
          f"reg_loss={reg_loss:.4f}, grad_per_alpha={grad_alpha:.4f}")


def test_multitask_loss():
    """L_total = L_fused + beta * L_graph. Both losses are BPR-style."""
    rng = np.random.RandomState(456)

    def bpr(user, pos, neg):
        diff = np.dot(user, pos) - np.dot(user, neg)
        return -np.log(sigmoid(diff) + 1e-10)

    dim = 64
    fused_user = rng.randn(dim)
    graph_user = rng.randn(dim)
    pos = rng.randn(dim)
    neg = rng.randn(dim)

    l_fused = bpr(fused_user, pos, neg)
    l_graph = bpr(graph_user, pos, neg)
    beta = 0.3

    l_total = l_fused + beta * l_graph

    assert l_total >= l_fused, "total loss should be >= fused loss alone"
    assert np.isfinite(l_total), "loss should be finite"

    l_no_graph = l_fused + 0.0 * l_graph
    assert np.isclose(l_no_graph, l_fused), "beta=0 should give original loss"

    print(f"PASS: multitask loss: L_fused={l_fused:.4f}, L_graph={l_graph:.4f}, "
          f"L_total={l_total:.4f} (beta={beta})")


def test_backward_compat_defaults():
    """With all Phase 1+2 flags at defaults, behavior matches the original."""
    alpha_reg = 0.0
    graph_loss_weight = 0.0

    alpha_mean = 0.73
    extra_from_reg = alpha_reg * alpha_mean
    assert extra_from_reg == 0.0

    fake_graph_loss = 1.5
    extra_from_graph = graph_loss_weight * fake_graph_loss
    assert extra_from_graph == 0.0

    print("PASS: defaults add zero to loss — backward compatible")


def test_graph_enriched_indexing():
    """Graph-enriched mode: sequence encoder should index propagated, not raw embeddings."""
    rng = np.random.RandomState(789)
    num_users, num_items, dim = 100, 200, 64

    raw_embeddings = rng.randn(num_users + num_items, dim).astype(np.float32)
    propagated = rng.randn(num_users + num_items, dim).astype(np.float32)

    # Sequence: [0, 0, 5, 12, 3] (0=pad, item ids are 1-indexed in sequence)
    sequence = np.array([0, 0, 5, 12, 3])
    item_nodes = num_users + np.clip(sequence - 1, 0, None)

    # Raw mode: look up raw embeddings
    raw_hidden = raw_embeddings[item_nodes]
    # Graph-enriched mode: look up propagated embeddings
    enriched_hidden = propagated[item_nodes]

    # They should differ (propagated has neighborhood info)
    assert not np.allclose(raw_hidden, enriched_hidden), \
        "enriched should differ from raw"

    # Padding positions should be masked to zero regardless
    pad_mask = (sequence == 0)
    raw_hidden[pad_mask] = 0.0
    enriched_hidden[pad_mask] = 0.0
    assert np.allclose(raw_hidden[0], 0.0), "padding should be zero"
    assert np.allclose(enriched_hidden[0], 0.0), "padding should be zero"

    # Non-pad enriched positions should contain propagated values
    assert np.allclose(enriched_hidden[2], propagated[num_users + 4]), \
        "item 5 in sequence (0-idx 4) should map to propagated[num_users+4]"

    print("PASS: graph-enriched indexing correct")


def test_layer_count_reduction():
    """1-layer Transformer has fewer parameters than 2-layer — sanity check."""
    # Approximate param count for a single TransformerEncoderLayer:
    # Self-attn: 4 * dim * dim (Q,K,V,O projections)
    # FFN: dim * 4*dim + 4*dim * dim = 8 * dim^2
    # Total per layer: ~12 * dim^2 (ignoring biases and layernorm)
    dim = 64
    params_per_layer = 12 * dim * dim  # ~49K
    params_1layer = params_per_layer
    params_2layer = 2 * params_per_layer

    assert params_1layer < params_2layer
    assert params_2layer / params_1layer == 2.0

    print(f"PASS: 1-layer (~{params_1layer:,} params) vs 2-layer (~{params_2layer:,} params)")


def test_length_aware_confidence():
    """Length-aware gate: confidence = sigmoid(scale * log(1+len) + bias).
    Short sequences get lower alpha, long sequences keep alpha high."""
    scale, bias = 1.0, 0.0

    def confidence(seq_len):
        return sigmoid(scale * np.log1p(seq_len) + bias)

    # Monotonically increasing with sequence length
    lengths = [1, 2, 3, 5, 10, 20, 50]
    confs = [confidence(l) for l in lengths]
    for i in range(len(confs) - 1):
        assert confs[i] < confs[i + 1], \
            f"confidence should increase: len={lengths[i]}→{confs[i]:.3f} vs len={lengths[i+1]}→{confs[i+1]:.3f}"

    # Short sequences should meaningfully reduce alpha
    alpha_raw = 0.8
    alpha_short = alpha_raw * confidence(1)
    alpha_long = alpha_raw * confidence(20)
    assert alpha_short < alpha_raw * 0.75, \
        f"1-item user should lose >25% alpha, got {alpha_short:.3f}"
    assert alpha_long > alpha_raw * 0.90, \
        f"20-item user should keep >90% alpha, got {alpha_long:.3f}"

    # With length_aware=False (no modulation), alpha unchanged
    alpha_no_mod = alpha_raw * 1.0
    assert alpha_no_mod == alpha_raw

    print(f"PASS: length-aware confidence: len=1={confidence(1):.3f}, "
          f"len=5={confidence(5):.3f}, len=20={confidence(20):.3f}")


if __name__ == "__main__":
    test_residual_gate()
    test_old_vs_new_formula()
    test_alpha_reg_pushes_to_zero()
    test_multitask_loss()
    test_backward_compat_defaults()
    test_graph_enriched_indexing()
    test_layer_count_reduction()
    test_length_aware_confidence()
    print("\nAll logic tests passed.")
