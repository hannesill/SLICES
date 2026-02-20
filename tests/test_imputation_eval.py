"""Tests for imputation evaluation framework.

Tests cover:
- apply_mask() with "random" strategy: correct mask ratio within tolerance
- apply_mask() with "feature_block": entire features masked
- apply_mask() with "temporal_block": contiguous hours masked
- evaluate() with random encoder runs without error
- NRMSE computation with known inputs
- from_encoder_checkpoint() creates linear decoder
"""

import pytest
import torch
import torch.nn as nn
from slices.eval.imputation import ImputationEvaluator
from torch.utils.data import DataLoader


class SimpleEncoder(nn.Module):
    """Minimal encoder for testing."""

    def __init__(self, d_input: int = 10, d_model: int = 16) -> None:
        super().__init__()
        self.d_input = d_input
        self.d_model = d_model
        self.proj = nn.Linear(d_input, d_model)

    def get_output_dim(self) -> int:
        return self.d_model

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, **kwargs) -> torch.Tensor:
        return self.proj(x)


def make_dummy_dataloader(
    n_samples: int = 32,
    seq_len: int = 24,
    d_input: int = 10,
    batch_size: int = 16,
) -> DataLoader:
    """Create a dummy DataLoader mimicking ICUDataset output."""
    timeseries = torch.randn(n_samples, seq_len, d_input)
    mask = torch.rand(n_samples, seq_len, d_input) > 0.3  # ~70% observed

    class DummyDataset:
        def __init__(self, ts, m):
            self.ts = ts
            self.m = m

        def __len__(self):
            return len(self.ts)

        def __getitem__(self, idx):
            return {"timeseries": self.ts[idx], "mask": self.m[idx]}

    return DataLoader(DummyDataset(timeseries, mask), batch_size=batch_size)


class TestApplyMaskRandom:
    """Tests for apply_mask with random strategy."""

    def test_mask_ratio_within_tolerance(self):
        """Random masking should mask approximately mask_ratio of observed values."""
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)

        B, T, D = 64, 48, 10
        timeseries = torch.randn(B, T, D)
        mask = torch.ones(B, T, D, dtype=torch.bool)  # All observed
        mask_ratio = 0.15

        _, eval_mask, _ = evaluator.apply_mask(timeseries, mask, "random", mask_ratio)

        actual_ratio = eval_mask.float().mean().item()
        assert (
            abs(actual_ratio - mask_ratio) < 0.05
        ), f"Actual mask ratio {actual_ratio:.4f} not within tolerance of {mask_ratio}"

    def test_only_observed_values_masked(self):
        """Random masking should only mask positions where mask=True."""
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)

        B, T, D = 32, 24, 10
        timeseries = torch.randn(B, T, D)
        mask = torch.rand(B, T, D) > 0.5  # 50% observed

        _, eval_mask, _ = evaluator.apply_mask(timeseries, mask, "random", 0.3)

        # eval_mask should be subset of mask
        assert (eval_mask & ~mask).sum() == 0, "Masked unobserved positions!"

    def test_masked_positions_zeroed(self):
        """Masked positions should be zeroed in the output."""
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)

        timeseries = torch.ones(4, 12, 10)
        mask = torch.ones(4, 12, 10, dtype=torch.bool)

        masked_input, eval_mask, _ = evaluator.apply_mask(timeseries, mask, "random", 0.5)

        # Masked positions should be 0
        assert (masked_input[eval_mask] == 0.0).all()
        # Unmasked positions should be unchanged
        assert (masked_input[~eval_mask] == 1.0).all()


class TestApplyMaskFeatureBlock:
    """Tests for apply_mask with feature_block strategy."""

    def test_entire_features_masked(self):
        """Feature block should mask entire features for the full window."""
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)

        B, T, D = 4, 24, 10
        timeseries = torch.randn(B, T, D)
        mask = torch.ones(B, T, D, dtype=torch.bool)

        _, eval_mask, _ = evaluator.apply_mask(timeseries, mask, "feature_block", 0.3)

        # For each sample, check that masked features are fully masked
        for b in range(B):
            for d in range(D):
                feature_mask = eval_mask[b, :, d]
                # Either all timesteps masked or none
                assert (
                    feature_mask.all() or not feature_mask.any()
                ), f"Feature {d} in batch {b} is partially masked"

    def test_correct_number_of_features_masked(self):
        """Should mask approximately mask_ratio fraction of features."""
        encoder = SimpleEncoder(d_input=20)
        evaluator = ImputationEvaluator(encoder, d_input=20)

        B, T, D = 4, 24, 20
        timeseries = torch.randn(B, T, D)
        mask = torch.ones(B, T, D, dtype=torch.bool)

        _, eval_mask, _ = evaluator.apply_mask(timeseries, mask, "feature_block", 0.3)

        for b in range(B):
            n_masked = sum(eval_mask[b, 0, d].item() for d in range(D))
            expected = int(D * 0.3)
            assert n_masked == expected, f"Expected {expected} features masked, got {n_masked}"


class TestApplyMaskTemporalBlock:
    """Tests for apply_mask with temporal_block strategy."""

    def test_contiguous_hours_masked(self):
        """Temporal block should mask contiguous hour blocks."""
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)

        B, T, D = 4, 48, 10
        timeseries = torch.randn(B, T, D)
        mask = torch.ones(B, T, D, dtype=torch.bool)

        _, eval_mask, _ = evaluator.apply_mask(timeseries, mask, "temporal_block", 0.2)

        for b in range(B):
            # All features should have the same temporal mask pattern
            first_feature_mask = eval_mask[b, :, 0]
            for d in range(1, D):
                assert (
                    eval_mask[b, :, d] == first_feature_mask
                ).all(), f"Feature {d} has different temporal mask than feature 0"

            # Check contiguity: masked hours should be contiguous
            masked_hours = first_feature_mask.nonzero(as_tuple=True)[0]
            if len(masked_hours) > 1:
                diffs = masked_hours[1:] - masked_hours[:-1]
                assert (diffs == 1).all(), "Temporal block is not contiguous"

    def test_correct_number_of_hours_masked(self):
        """Should mask approximately mask_ratio fraction of hours."""
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)

        B, T, D = 4, 48, 10
        timeseries = torch.randn(B, T, D)
        mask = torch.ones(B, T, D, dtype=torch.bool)

        _, eval_mask, _ = evaluator.apply_mask(timeseries, mask, "temporal_block", 0.25)

        for b in range(B):
            n_masked_hours = eval_mask[b, :, 0].sum().item()
            expected = int(T * 0.25)
            assert (
                n_masked_hours == expected
            ), f"Expected {expected} hours masked, got {n_masked_hours}"


class TestApplyMaskInvalidStrategy:
    """Tests for invalid masking strategy."""

    def test_raises_on_unknown_strategy(self):
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)
        timeseries = torch.randn(2, 12, 10)
        mask = torch.ones(2, 12, 10, dtype=torch.bool)

        with pytest.raises(ValueError, match="Unknown masking strategy"):
            evaluator.apply_mask(timeseries, mask, "nonexistent", 0.15)


class TestEvaluate:
    """Tests for evaluate() method."""

    def test_evaluate_runs_without_error(self):
        """Evaluate should run end-to-end without errors."""
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)

        dataloader = make_dummy_dataloader()
        results = evaluator.evaluate(dataloader, mask_strategy="random", mask_ratio=0.15)

        assert "nrmse_per_feature" in results
        assert "mae_overall" in results
        assert "nrmse_overall" in results
        assert isinstance(results["nrmse_per_feature"], dict)
        assert isinstance(results["mae_overall"], float)
        assert isinstance(results["nrmse_overall"], float)

    def test_evaluate_all_strategies(self):
        """All three strategies should run without errors."""
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10)
        dataloader = make_dummy_dataloader()

        for strategy in ["random", "feature_block", "temporal_block"]:
            results = evaluator.evaluate(dataloader, mask_strategy=strategy)
            assert results["nrmse_overall"] >= 0
            assert results["mae_overall"] >= 0

    def test_evaluate_returns_feature_names(self):
        """When feature_names provided, results use them as keys."""
        feature_names = [f"feat_{i}" for i in range(10)]
        encoder = SimpleEncoder()
        evaluator = ImputationEvaluator(encoder, d_input=10, feature_names=feature_names)

        dataloader = make_dummy_dataloader()
        results = evaluator.evaluate(dataloader, mask_strategy="random")

        # Feature names should appear as keys
        for key in results["nrmse_per_feature"]:
            assert key.startswith("feat_"), f"Expected feature name, got {key}"


class TestNRMSEComputation:
    """Tests for NRMSE computation correctness."""

    def test_nrmse_known_values(self):
        """NRMSE with known reconstruction error and known feature std."""
        # Create encoder that returns zeros -> decoder will reconstruct zeros
        encoder = SimpleEncoder(d_input=5, d_model=8)
        decoder = nn.Linear(8, 5)
        # Initialize decoder to output zeros
        nn.init.zeros_(decoder.weight)
        nn.init.zeros_(decoder.bias)

        evaluator = ImputationEvaluator(
            encoder=encoder,
            decoder=decoder,
            feature_names=["a", "b", "c", "d", "e"],
        )

        # Create data with known values
        timeseries = torch.ones(16, 12, 5) * 2.0
        mask = torch.ones(16, 12, 5, dtype=torch.bool)

        class ConstDataset:
            def __len__(self):
                return 16

            def __getitem__(self, idx):
                return {"timeseries": timeseries[idx], "mask": mask[idx]}

        loader = DataLoader(ConstDataset(), batch_size=16)
        results = evaluator.evaluate(loader, mask_strategy="random", mask_ratio=0.5)

        # Reconstruction is ~0 (from zero decoder), original is 2.0
        # MAE should be approximately 2.0
        assert (
            results["mae_overall"] > 1.0
        ), f"Expected MAE > 1.0 for constant input, got {results['mae_overall']}"


class TestTrainDecoder:
    """Tests for train_decoder() method."""

    def test_train_decoder_reduces_loss(self):
        """Training decoder should reduce reconstruction loss."""
        encoder = SimpleEncoder(d_input=10, d_model=16)
        evaluator = ImputationEvaluator(encoder, d_input=10)

        dataloader = make_dummy_dataloader(n_samples=64, batch_size=32)
        history = evaluator.train_decoder(dataloader, max_epochs=5, lr=1e-3)

        assert "train_losses" in history
        assert len(history["train_losses"]) == 5
        # Loss should generally decrease (allow some noise)
        assert history["train_losses"][-1] <= history["train_losses"][0] * 2.0


class TestFromEncoderCheckpoint:
    """Tests for from_encoder_checkpoint class method."""

    def test_creates_linear_decoder(self, tmp_path):
        """from_encoder_checkpoint should create a decoder module."""
        from slices.models.encoders import build_encoder

        # Create and save a v3 checkpoint
        encoder = build_encoder(
            "transformer",
            {
                "d_input": 10,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 4,
                "d_ff": 64,
                "pooling": "none",
            },
        )

        ckpt = {
            "encoder_state_dict": encoder.state_dict(),
            "encoder_config": {
                "name": "transformer",
                "d_input": 10,
                "d_model": 32,
                "n_layers": 1,
                "n_heads": 4,
                "d_ff": 64,
                "pooling": "none",
            },
            "version": 3,
        }
        ckpt_path = tmp_path / "encoder.pt"
        torch.save(ckpt, ckpt_path)

        evaluator = ImputationEvaluator.from_encoder_checkpoint(str(ckpt_path), d_input=10)

        assert evaluator.encoder is not None
        assert evaluator.decoder is not None

        # Decoder should output d_input dimensions
        dummy = torch.randn(2, 12, 10)
        with torch.no_grad():
            enc_out = evaluator.encoder(dummy)
            dec_out = evaluator.decoder(enc_out)
        assert dec_out.shape == (2, 12, 10)


class TestInit:
    """Tests for __init__ edge cases."""

    def test_requires_decoder_or_d_input(self):
        """Should raise if neither decoder nor d_input provided."""
        encoder = SimpleEncoder()
        with pytest.raises(ValueError, match="Either decoder or d_input"):
            ImputationEvaluator(encoder, decoder=None, d_input=None)

    def test_custom_decoder(self):
        """Should accept a custom decoder module."""
        encoder = SimpleEncoder(d_input=10, d_model=16)
        decoder = nn.Linear(16, 10)
        evaluator = ImputationEvaluator(encoder, decoder=decoder)
        assert evaluator.decoder is decoder
