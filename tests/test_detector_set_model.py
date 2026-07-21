from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gwyolo.numeric import (  # noqa: E402
    DetectorSetGlitchEmbeddingNet,
    DetectorSetQNet,
    GlitchEmbeddingNet,
    MultiIFOQNet,
    initialize_detector_set_from_early_fusion,
    model_from_checkpoint,
)


def test_detector_set_model_masks_missing_ifo_and_backpropagates() -> None:
    model = DetectorSetQNet(ifo_count=3, q_count=2, base_channels=8)
    features = torch.randn(2, 6, 16, 12, requires_grad=True)
    availability = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32)
    logits = model(features, availability)
    assert logits.shape == (2, 2, 6, 16, 12)
    reshaped = logits.reshape(2, 2, 3, 2, 16, 12)
    assert torch.all(reshaped[0, :, 2] == -20)
    assert torch.all(reshaped[1, :, 1] == -20)
    logits.sum().backward()
    assert model.shared_encoder.layers[0].weight.grad is not None


def test_detector_set_fusion_is_equivariant_to_ifo_slot_permutation() -> None:
    torch.manual_seed(1)
    model = DetectorSetQNet(ifo_count=3, q_count=1, base_channels=8).eval()
    features = torch.randn(1, 3, 8, 8)
    availability = torch.ones(1, 3)
    baseline = model(features, availability).reshape(1, 2, 3, 1, 8, 8)
    permutation = [2, 0, 1]
    permuted = model(features[:, permutation], availability[:, permutation]).reshape(
        1, 2, 3, 1, 8, 8
    )
    assert torch.allclose(permuted, baseline[:, :, permutation], atol=1e-6, rtol=1e-6)


def test_detector_set_warm_start_mapping_is_hand_calculated() -> None:
    source = MultiIFOQNet(input_channels=6, base_channels=8)
    with torch.no_grad():
        first = source.encoder.layers[0].weight
        first.copy_(torch.arange(first.numel(), dtype=first.dtype).reshape_as(first))
        source.head.weight.copy_(
            torch.arange(source.head.weight.numel(), dtype=source.head.weight.dtype).reshape_as(
                source.head.weight
            )
        )
    target = DetectorSetQNet(ifo_count=3, q_count=2, base_channels=8)
    report = initialize_detector_set_from_early_fusion(target, source.state_dict())
    expected_first = source.encoder.layers[0].weight.reshape(8, 3, 2, 3, 3).mean(dim=1)
    expected_head = source.head.weight.reshape(2, 3, 2, 8, 1, 1).mean(dim=1).reshape(
        4, 8, 1, 1
    )
    assert torch.equal(target.shared_encoder.layers[0].weight, expected_first)
    assert torch.equal(target.shared_head.weight, expected_head)
    assert torch.count_nonzero(target.attention_score.weight) == 0
    assert report["input_channels"] == 6


def test_detector_set_rejects_implicit_or_empty_availability() -> None:
    model = DetectorSetQNet(ifo_count=2, q_count=1, base_channels=8)
    features = torch.zeros(1, 2, 8, 8)
    with pytest.raises(ValueError, match="at least one available"):
        model(features, torch.zeros(1, 2))
    with pytest.raises(ValueError, match="shape"):
        model(features, torch.ones(1, 3))


def test_checkpoint_loader_preserves_architecture_and_detector_order() -> None:
    source = DetectorSetQNet(ifo_count=3, q_count=1, base_channels=8)
    checkpoint = {
        "architecture": "detector_set",
        "input_channels": 3,
        "base_channels": 8,
        "model_ifos": ["H1", "L1", "V1"],
        "q_values": [4.0],
        "model": source.state_dict(),
    }
    restored, architecture = model_from_checkpoint(
        checkpoint, ("H1", "L1", "V1"), (4.0,)
    )
    assert isinstance(restored, DetectorSetQNet)
    assert architecture == "detector_set"
    with pytest.raises(ValueError, match="detector ordering"):
        model_from_checkpoint(checkpoint, ("L1", "H1", "V1"), (4.0,))


def test_glitch_embedding_is_normalized_and_classifies_known_families() -> None:
    model = GlitchEmbeddingNet(q_count=3, class_count=4, base_channels=8, embedding_dim=6)
    logits, embedding = model(torch.randn(5, 3, 12, 10))
    assert logits.shape == (5, 4)
    assert embedding.shape == (5, 6)
    assert torch.allclose(torch.linalg.norm(embedding, dim=1), torch.ones(5), atol=1e-6)


def test_detector_set_glitch_embedding_masks_missing_ifos_and_keeps_identity() -> None:
    torch.manual_seed(7)
    model = DetectorSetGlitchEmbeddingNet(
        ifo_count=3, q_count=2, class_count=4, base_channels=8, embedding_dim=6
    ).eval()
    features = torch.randn(2, 3, 2, 12, 10)
    availability = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.float32)
    logits, embedding = model(features, availability)
    changed = features.clone()
    changed[0, 2] = 1_000.0
    changed[1, 1] = -1_000.0
    changed_logits, changed_embedding = model(changed, availability)
    assert logits.shape == (2, 4)
    assert embedding.shape == (2, 6)
    assert torch.allclose(torch.linalg.norm(embedding, dim=1), torch.ones(2), atol=1e-6)
    # Unavailable planes are mathematically excluded, even if a caller supplies garbage.
    assert torch.allclose(changed_logits, logits, atol=1e-6, rtol=1e-6)
    assert torch.allclose(changed_embedding, embedding, atol=1e-6, rtol=1e-6)
    # Fixed one-hot detector identities are part of the projection input.
    assert model.network_channels == 2 * model.base_channels + model.ifo_count


def test_detector_set_glitch_embedding_rejects_implicit_availability() -> None:
    model = DetectorSetGlitchEmbeddingNet(
        ifo_count=3, q_count=1, class_count=2, base_channels=8, embedding_dim=4
    )
    features = torch.zeros(1, 3, 1, 8, 8)
    with pytest.raises(ValueError, match="at least one available"):
        model(features, torch.zeros(1, 3))
    with pytest.raises(ValueError, match="shape"):
        model(features, torch.ones(1, 2))
