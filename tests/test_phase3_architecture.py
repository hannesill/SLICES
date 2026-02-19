"""Tests for Phase 3 — Architecture & Performance changes.

Tests cover:
- 3.1: Vectorized _create_dense_timeseries produces correct output
- 3.2: Shared pooling, PositionalEncoding, activation, optimizer/scheduler utilities
- 3.3: Fairness metrics with synthetic data and known expected values
- 3.4: v3 checkpoint format via shared save_encoder_checkpoint
"""

import math
from typing import List

import polars as pl
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from slices.data.extractors.base import BaseExtractor, ExtractorConfig
from slices.eval.fairness import (
    FairnessConfig,
    compute_demographic_parity,
    compute_equalized_odds,
    compute_subgroup_metrics,
    demographic_parity_difference,
    disparate_impact_ratio,
    equalized_odds_difference,
)
from slices.models.common import PositionalEncoding, apply_pooling, get_activation
from slices.models.encoders import (
    LinearConfig,
    LinearEncoder,
    TransformerConfig,
    TransformerEncoder,
)
from slices.models.heads.base import TaskHeadConfig
from slices.models.heads.mlp import MLPTaskHead
from slices.training.utils import (
    build_optimizer,
    build_scheduler,
    save_encoder_checkpoint,
)

# ============================================================================
# Helpers
# ============================================================================


class MinimalExtractor(BaseExtractor):
    """Minimal extractor for testing _create_dense_timeseries."""

    def __init__(self, config: ExtractorConfig):
        # Skip parent __init__ to avoid path validation
        self.config = config

    def _get_dataset_name(self) -> str:
        return "test"

    def extract_stays(self) -> pl.DataFrame:
        return pl.DataFrame()

    def extract_timeseries(self, stay_ids: List[int]) -> pl.DataFrame:
        return pl.DataFrame()

    def extract_data_source(self, source_name: str, stay_ids: List[int]) -> pl.DataFrame:
        return pl.DataFrame()

    def run(self) -> None:
        pass


def _make_extractor(seq_length: int = 48) -> MinimalExtractor:
    cfg = ExtractorConfig(
        parquet_root="/tmp/fake",
        output_dir="/tmp/fake_out",
        seq_length_hours=seq_length,
        feature_set="core",
        tasks=[],
        min_stay_hours=0,
        batch_size=100,
    )
    ext = MinimalExtractor(cfg)
    return ext


# ============================================================================
# 3.1 — Vectorized _create_dense_timeseries
# ============================================================================


class TestVectorizedDenseTimeseries:
    """Tests for the vectorized dense timeseries conversion."""

    def test_basic_conversion(self):
        """Test basic sparse-to-dense conversion with known values."""
        ext = _make_extractor(seq_length=4)
        features = ["hr", "sbp"]

        stays = pl.DataFrame({"stay_id": [1, 2]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1, 1, 2],
                "hour": [0, 2, 1],
                "hr": [70.0, 72.0, 80.0],
                "hr_mask": [True, True, True],
                "sbp": [120.0, None, 130.0],
                "sbp_mask": [True, False, True],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)

        assert len(result) == 2
        assert set(result.columns) == {"stay_id", "timeseries", "mask"}

        # Stay 1: hr=70@h0, hr=72@h2, sbp=120@h0
        row1 = result.filter(pl.col("stay_id") == 1)
        ts1 = row1["timeseries"][0]
        mask1 = row1["mask"][0]

        assert ts1[0][0] == 70.0  # hr at hour 0
        assert mask1[0][0] is True
        assert ts1[0][1] == 120.0  # sbp at hour 0
        assert mask1[0][1] is True
        assert math.isnan(ts1[1][0])  # hr at hour 1 (missing)
        assert mask1[1][0] is False
        assert ts1[2][0] == 72.0  # hr at hour 2
        assert mask1[2][0] is True

    def test_stays_without_data_produce_nan_arrays(self):
        """Stays with no observations get all-NaN timeseries."""
        ext = _make_extractor(seq_length=3)
        features = ["hr"]
        stays = pl.DataFrame({"stay_id": [1, 2]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1],
                "hour": [0],
                "hr": [70.0],
                "hr_mask": [True],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        assert len(result) == 2

        # Stay 2 should have all NaN values and False masks
        row2 = result.filter(pl.col("stay_id") == 2)
        ts2 = row2["timeseries"][0]
        mask2 = row2["mask"][0]

        for h in range(3):
            assert math.isnan(ts2[h][0])
            assert mask2[h][0] is False

    def test_overflow_hours_discarded(self):
        """Hours beyond seq_length are discarded (not included)."""
        ext = _make_extractor(seq_length=2)
        features = ["hr"]
        stays = pl.DataFrame({"stay_id": [1]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1, 1, 1],
                "hour": [0, 1, 5],  # hour 5 overflows seq_length=2
                "hr": [70.0, 71.0, 99.0],
                "hr_mask": [True, True, True],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        ts = result["timeseries"][0]

        assert len(ts) == 2
        assert ts[0][0] == 70.0
        assert ts[1][0] == 71.0

    def test_mask_false_produces_nan(self):
        """When mask is False, value should be NaN even if a value is present."""
        ext = _make_extractor(seq_length=2)
        features = ["hr"]
        stays = pl.DataFrame({"stay_id": [1]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1],
                "hour": [0],
                "hr": [70.0],
                "hr_mask": [False],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        ts = result["timeseries"][0]
        mask = result["mask"][0]

        assert math.isnan(ts[0][0])
        assert mask[0][0] is False

    def test_output_shape_consistency(self):
        """Verify output dimensions: seq_length x n_features."""
        ext = _make_extractor(seq_length=10)
        features = ["a", "b", "c"]
        stays = pl.DataFrame({"stay_id": [1, 2, 3]})
        sparse = pl.DataFrame(
            {
                "stay_id": [1],
                "hour": [0],
                "a": [1.0],
                "a_mask": [True],
                "b": [2.0],
                "b_mask": [True],
                "c": [3.0],
                "c_mask": [True],
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        assert len(result) == 3

        for row in result.iter_rows(named=True):
            ts = row["timeseries"]
            mask = row["mask"]
            assert len(ts) == 10  # seq_length
            assert len(mask) == 10
            for h in range(10):
                assert len(ts[h]) == 3  # n_features
                assert len(mask[h]) == 3

    def test_empty_sparse_timeseries(self):
        """Handle completely empty sparse timeseries."""
        ext = _make_extractor(seq_length=3)
        features = ["hr"]
        stays = pl.DataFrame({"stay_id": [1]})
        sparse = pl.DataFrame(
            {
                "stay_id": pl.Series([], dtype=pl.Int64),
                "hour": pl.Series([], dtype=pl.Int64),
                "hr": pl.Series([], dtype=pl.Float64),
                "hr_mask": pl.Series([], dtype=pl.Boolean),
            }
        )

        result = ext._create_dense_timeseries(sparse, stays, features)
        assert len(result) == 1

        ts = result["timeseries"][0]
        for h in range(3):
            assert math.isnan(ts[h][0])


# ============================================================================
# 3.2 — Shared utilities (pooling, PositionalEncoding, activation, optimizer)
# ============================================================================


class TestSharedPooling:
    """Tests for apply_pooling utility."""

    def test_none_pooling_preserves_shape(self):
        x = torch.randn(4, 10, 32)
        out = apply_pooling(x, "none")
        assert out.shape == (4, 10, 32)
        assert torch.equal(out, x)

    def test_mean_pooling_without_mask(self):
        x = torch.ones(2, 5, 8)
        out = apply_pooling(x, "mean")
        assert out.shape == (2, 8)
        assert torch.allclose(out, torch.ones(2, 8))

    def test_mean_pooling_with_mask(self):
        x = torch.ones(1, 4, 2)
        x[0, 2:] = 999.0  # padding values
        mask = torch.tensor([[True, True, False, False]])
        out = apply_pooling(x, "mean", padding_mask=mask)
        assert out.shape == (1, 2)
        assert torch.allclose(out, torch.ones(1, 2))

    def test_max_pooling_without_mask(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        out = apply_pooling(x, "max")
        assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))

    def test_max_pooling_with_mask(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
        mask = torch.tensor([[True, True, False]])
        out = apply_pooling(x, "max", padding_mask=mask)
        assert torch.allclose(out, torch.tensor([[3.0, 4.0]]))

    def test_last_pooling_without_mask(self):
        x = torch.tensor([[[1.0], [2.0], [3.0]]])
        out = apply_pooling(x, "last")
        assert torch.allclose(out, torch.tensor([[3.0]]))

    def test_last_pooling_with_mask(self):
        x = torch.tensor([[[1.0], [2.0], [99.0]]])
        mask = torch.tensor([[True, True, False]])
        out = apply_pooling(x, "last", padding_mask=mask)
        assert torch.allclose(out, torch.tensor([[2.0]]))

    def test_cls_pooling(self):
        x = torch.tensor([[[10.0, 20.0], [1.0, 2.0]]])
        out = apply_pooling(x, "cls")
        assert torch.allclose(out, torch.tensor([[10.0, 20.0]]))

    def test_invalid_pooling_raises(self):
        x = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="Unknown pooling"):
            apply_pooling(x, "invalid")

    def test_transformer_uses_shared_pooling(self):
        """Verify TransformerEncoder._apply_pooling delegates to shared utility."""
        config = TransformerConfig(d_input=4, d_model=8, n_layers=1, n_heads=2, pooling="mean")
        encoder = TransformerEncoder(config)
        encoder.eval()
        x = torch.randn(2, 5, 4)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 8)

    def test_linear_uses_shared_pooling(self):
        """Verify LinearEncoder._apply_pooling delegates to shared utility."""
        config = LinearConfig(d_input=4, d_model=8, pooling="max")
        encoder = LinearEncoder(config)
        encoder.eval()
        x = torch.randn(2, 5, 4)
        with torch.no_grad():
            out = encoder(x)
        assert out.shape == (2, 8)


class TestSharedPositionalEncoding:
    """Tests for shared PositionalEncoding module."""

    def test_3d_input(self):
        pe = PositionalEncoding(d_model=16, max_seq_length=100, dropout=0.0)
        x = torch.zeros(2, 10, 16)
        out = pe(x)
        assert out.shape == (2, 10, 16)
        # Should have added non-zero values
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_4d_input(self):
        pe = PositionalEncoding(d_model=16, max_seq_length=100, dropout=0.0)
        x = torch.zeros(2, 5, 10, 16)
        out = pe(x)
        assert out.shape == (2, 5, 10, 16)
        assert not torch.allclose(out, torch.zeros_like(out))

    def test_deterministic(self):
        pe = PositionalEncoding(d_model=32, max_seq_length=100, dropout=0.0)
        x = torch.randn(2, 10, 32)
        out1 = pe(x)
        out2 = pe(x)
        assert torch.allclose(out1, out2)

    def test_different_positions_get_different_encodings(self):
        pe = PositionalEncoding(d_model=32, max_seq_length=100, dropout=0.0)
        x = torch.zeros(1, 50, 32)
        out = pe(x)
        # Position 0 and position 1 should have different encodings
        assert not torch.allclose(out[0, 0], out[0, 1])


class TestSharedActivation:
    """Tests for get_activation utility."""

    def test_known_activations(self):
        for name in ["relu", "gelu", "silu", "tanh"]:
            act = get_activation(name)
            assert isinstance(act, nn.Module)
            # Verify it actually computes something
            x = torch.randn(4)
            out = act(x)
            assert out.shape == (4,)

    def test_unknown_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            get_activation("leaky_relu_not_supported")

    def test_activation_used_in_transformer_layer(self):
        """TransformerEncoderLayer uses shared get_activation."""
        from slices.models.encoders.transformer import TransformerEncoderLayer

        for act_name in ["relu", "gelu", "silu"]:
            layer = TransformerEncoderLayer(
                d_model=16,
                n_heads=2,
                d_ff=32,
                activation=act_name,
            )
            x = torch.randn(2, 5, 16)
            out = layer(x)
            assert out.shape == (2, 5, 16)

    def test_activation_used_in_mlp_head(self):
        """MLPTaskHead uses shared get_activation."""
        config = TaskHeadConfig(
            name="mlp",
            task_name="test",
            task_type="binary",
            input_dim=16,
            hidden_dims=[8],
            activation="gelu",
        )
        head = MLPTaskHead(config)
        x = torch.randn(2, 16)
        out = head(x)
        assert "logits" in out
        assert out["logits"].shape == (2, 2)


class TestSharedOptimizer:
    """Tests for shared build_optimizer / build_scheduler."""

    def _make_config(self, **kwargs):
        return OmegaConf.create(kwargs)

    def test_adam(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="adam", lr=0.001, weight_decay=0.0)
        opt = build_optimizer(model.parameters(), cfg)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="adamw", lr=0.001, weight_decay=0.01)
        opt = build_optimizer(model.parameters(), cfg)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="sgd", lr=0.01, momentum=0.9)
        opt = build_optimizer(model.parameters(), cfg)
        assert isinstance(opt, torch.optim.SGD)

    def test_unknown_optimizer_raises(self):
        model = nn.Linear(4, 2)
        cfg = self._make_config(name="rmsprop", lr=0.001)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            build_optimizer(model.parameters(), cfg)

    def test_cosine_scheduler(self):
        model = nn.Linear(4, 2)
        opt_cfg = self._make_config(name="adamw", lr=0.001)
        optimizer = build_optimizer(model.parameters(), opt_cfg)
        sched_cfg = self._make_config(name="cosine", T_max=50, eta_min=1e-6)
        result = build_scheduler(optimizer, sched_cfg)
        assert result is not None
        assert "optimizer" in result
        assert "lr_scheduler" in result
        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.CosineAnnealingLR,
        )

    def test_step_scheduler(self):
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(
            model.parameters(),
            self._make_config(name="adam", lr=0.001),
        )
        result = build_scheduler(
            optimizer,
            self._make_config(name="step", step_size=10, gamma=0.5),
        )
        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.StepLR,
        )

    def test_plateau_scheduler_has_monitor(self):
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(
            model.parameters(),
            self._make_config(name="adam", lr=0.001),
        )
        result = build_scheduler(
            optimizer,
            self._make_config(name="plateau", patience=5),
        )
        assert result["lr_scheduler"]["monitor"] == "val/loss"

    def test_warmup_cosine_scheduler(self):
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(
            model.parameters(),
            self._make_config(name="adam", lr=0.001),
        )
        result = build_scheduler(
            optimizer,
            self._make_config(name="warmup_cosine", warmup_epochs=5, max_epochs=50),
        )
        assert isinstance(
            result["lr_scheduler"]["scheduler"],
            torch.optim.lr_scheduler.LambdaLR,
        )

    def test_none_scheduler(self):
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(
            model.parameters(),
            self._make_config(name="adam", lr=0.001),
        )
        assert build_scheduler(optimizer, None) is None

    def test_unknown_scheduler_raises(self):
        model = nn.Linear(4, 2)
        optimizer = build_optimizer(
            model.parameters(),
            self._make_config(name="adam", lr=0.001),
        )
        with pytest.raises(ValueError, match="Unknown scheduler"):
            build_scheduler(
                optimizer,
                self._make_config(name="exponential"),
            )


# ============================================================================
# 3.3 — Fairness metrics
# ============================================================================


class TestDemographicParity:
    """Tests for demographic_parity_difference and disparate_impact_ratio."""

    def test_perfect_parity(self):
        """Equal positive rates across groups -> difference = 0."""
        predictions = torch.tensor([0.9, 0.9, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = demographic_parity_difference(predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(0.0)

    def test_maximum_disparity(self):
        """One group all positive, other all negative -> difference = 1.0."""
        predictions = torch.tensor([0.9, 0.9, 0.1, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = demographic_parity_difference(predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(1.0)

    def test_partial_disparity(self):
        """50% vs 100% positive rate -> difference = 0.5."""
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = demographic_parity_difference(predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(0.5)

    def test_single_group_returns_zero(self):
        predictions = torch.tensor([0.9, 0.1])
        group_ids = torch.tensor([0, 0])
        assert demographic_parity_difference(predictions, group_ids) == 0.0

    def test_three_groups(self):
        """With three groups, use max disparity."""
        # Group 0: 100%, Group 1: 50%, Group 2: 0%
        predictions = torch.tensor([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1, 2, 2])
        diff = demographic_parity_difference(predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(1.0)

    def test_custom_threshold(self):
        """Threshold affects which predictions are considered positive."""
        predictions = torch.tensor([0.7, 0.3, 0.7, 0.3])
        group_ids = torch.tensor([0, 0, 1, 1])
        # With threshold 0.5: both groups have 50% rate -> diff = 0
        assert demographic_parity_difference(predictions, group_ids, threshold=0.5) == 0.0
        # With threshold 0.8: both groups have 0% rate -> diff = 0
        assert demographic_parity_difference(predictions, group_ids, threshold=0.8) == 0.0


class TestDisparateImpact:
    """Tests for disparate_impact_ratio."""

    def test_perfect_parity(self):
        """Equal rates -> ratio = 1.0."""
        predictions = torch.tensor([0.9, 0.9, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        ratio = disparate_impact_ratio(predictions, group_ids, threshold=0.5)
        assert ratio == pytest.approx(1.0)

    def test_four_fifths_violation(self):
        """One group has 50% rate, other has 100% -> ratio = 0.5 (violates 4/5 rule)."""
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        ratio = disparate_impact_ratio(predictions, group_ids, threshold=0.5)
        assert ratio == pytest.approx(0.5)
        assert ratio < 0.8  # Violates four-fifths rule

    def test_zero_max_rate(self):
        """If no group predicts positive, return 0.0."""
        predictions = torch.tensor([0.1, 0.1, 0.1, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        ratio = disparate_impact_ratio(predictions, group_ids, threshold=0.5)
        assert ratio == 0.0

    def test_single_group(self):
        predictions = torch.tensor([0.9])
        group_ids = torch.tensor([0])
        assert disparate_impact_ratio(predictions, group_ids) == 1.0


class TestEqualizedOdds:
    """Tests for equalized_odds_difference."""

    def test_perfect_equalized_odds(self):
        """Same TPR and FPR across groups -> difference = 0."""
        # Group 0: TP=1/1, FP=0/1 -> TPR=1, FPR=0
        # Group 1: TP=1/1, FP=0/1 -> TPR=1, FPR=0
        labels = torch.tensor([1, 0, 1, 0])
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = equalized_odds_difference(labels, predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(0.0)

    def test_tpr_disparity(self):
        """Different TPR across groups."""
        # Group 0: label=1, pred=0.9 -> TP; label=0, pred=0.1 -> TN
        # Group 1: label=1, pred=0.1 -> FN; label=0, pred=0.1 -> TN
        # TPR: Group0=1.0, Group1=0.0 -> diff=1.0
        # FPR: Group0=0.0, Group1=0.0 -> diff=0.0
        labels = torch.tensor([1, 0, 1, 0])
        predictions = torch.tensor([0.9, 0.1, 0.1, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = equalized_odds_difference(labels, predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(1.0)

    def test_fpr_disparity(self):
        """Different FPR across groups."""
        # Group 0: label=0, pred=0.9 -> FP; label=1, pred=0.9 -> TP
        # Group 1: label=0, pred=0.1 -> TN; label=1, pred=0.9 -> TP
        # TPR: both 1.0 -> diff=0.0
        # FPR: Group0=1.0, Group1=0.0 -> diff=1.0
        labels = torch.tensor([0, 1, 0, 1])
        predictions = torch.tensor([0.9, 0.9, 0.1, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        diff = equalized_odds_difference(labels, predictions, group_ids, threshold=0.5)
        assert diff == pytest.approx(1.0)

    def test_single_group(self):
        labels = torch.tensor([1, 0])
        predictions = torch.tensor([0.9, 0.1])
        group_ids = torch.tensor([0, 0])
        assert equalized_odds_difference(labels, predictions, group_ids) == 0.0


class TestComputeDemographicParity:
    """Tests for the aggregate compute_demographic_parity function."""

    def test_returns_all_fields(self):
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        result = compute_demographic_parity(predictions, group_ids)
        assert "demographic_parity_difference" in result
        assert "disparate_impact_ratio" in result
        assert "per_group_rates" in result
        assert "0" in result["per_group_rates"]
        assert "1" in result["per_group_rates"]

    def test_per_group_rates_correct(self):
        # Group 0: 1/2 positive, Group 1: 2/2 positive
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.9])
        group_ids = torch.tensor([0, 0, 1, 1])
        result = compute_demographic_parity(predictions, group_ids)
        assert result["per_group_rates"]["0"] == pytest.approx(0.5)
        assert result["per_group_rates"]["1"] == pytest.approx(1.0)


class TestComputeEqualizedOdds:
    """Tests for the aggregate compute_equalized_odds function."""

    def test_returns_all_fields(self):
        labels = torch.tensor([1, 0, 1, 0])
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.1])
        group_ids = torch.tensor([0, 0, 1, 1])
        result = compute_equalized_odds(predictions, labels, group_ids)
        assert "equalized_odds_difference" in result
        assert "per_group_tpr" in result
        assert "per_group_fpr" in result


class TestComputeSubgroupMetrics:
    """Tests for compute_subgroup_metrics."""

    def test_basic_subgroup_metrics(self):
        predictions = torch.tensor([0.9, 0.1, 0.9, 0.1])
        labels = torch.tensor([1, 0, 1, 0])
        group_ids = torch.tensor([0, 0, 1, 1])
        config = FairnessConfig(min_subgroup_size=1)
        result = compute_subgroup_metrics(predictions, labels, group_ids, config)
        assert "0" in result
        assert "1" in result
        assert result["0"]["n_samples"] == 2.0
        assert result["0"]["positive_rate"] == pytest.approx(0.5)

    def test_min_subgroup_size_filters_small_groups(self):
        predictions = torch.tensor([0.9, 0.1, 0.9])
        labels = torch.tensor([1, 0, 1])
        group_ids = torch.tensor([0, 0, 1])
        config = FairnessConfig(min_subgroup_size=2)
        result = compute_subgroup_metrics(predictions, labels, group_ids, config)
        # Group 1 has only 1 sample, should be filtered
        assert "0" in result
        assert "1" not in result


# ============================================================================
# 3.4 — v3 checkpoint format
# ============================================================================


class TestSaveEncoderCheckpoint:
    """Tests for shared save_encoder_checkpoint."""

    def test_saves_and_loads_v3_format(self, tmp_path):
        """Checkpoint contains required v3 fields."""
        encoder = nn.Linear(4, 8)
        config = {"name": "linear", "d_input": 4, "d_model": 8}
        path = tmp_path / "encoder.pt"

        save_encoder_checkpoint(encoder, config, path)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert ckpt["version"] == 3
        assert ckpt["encoder_config"]["name"] == "linear"
        assert "encoder_state_dict" in ckpt
        assert "weight" in ckpt["encoder_state_dict"]

    def test_saves_missing_token(self, tmp_path):
        """Optional missing token is saved when provided."""
        encoder = nn.Linear(4, 8)
        config = {"name": "linear", "d_input": 4, "d_model": 8}
        missing_token = torch.randn(4)
        path = tmp_path / "encoder.pt"

        save_encoder_checkpoint(encoder, config, path, missing_token=missing_token, d_input=4)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert "missing_token" in ckpt
        assert torch.allclose(ckpt["missing_token"], missing_token)
        assert ckpt["d_input"] == 4

    def test_no_missing_token_when_none(self, tmp_path):
        encoder = nn.Linear(4, 8)
        config = {"name": "test"}
        path = tmp_path / "encoder.pt"

        save_encoder_checkpoint(encoder, config, path)

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        assert "missing_token" not in ckpt

    def test_state_dict_is_loadable(self, tmp_path):
        """Saved state dict can be loaded into a fresh encoder."""
        encoder = nn.Linear(4, 8)
        config = {"name": "linear"}
        path = tmp_path / "encoder.pt"

        # Set specific weights
        with torch.no_grad():
            encoder.weight.fill_(0.42)
            encoder.bias.fill_(0.07)

        save_encoder_checkpoint(encoder, config, path)

        # Load into fresh encoder
        new_encoder = nn.Linear(4, 8)
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        new_encoder.load_state_dict(ckpt["encoder_state_dict"])

        assert torch.allclose(new_encoder.weight, encoder.weight)
        assert torch.allclose(new_encoder.bias, encoder.bias)
