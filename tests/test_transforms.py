"""Tests for slices/data/transforms.py.

Tests SSL masking strategies and data augmentation functions for self-supervised
learning on ICU time-series data.
"""

import pytest
import torch
from slices.data.transforms import (
    apply_gaussian_noise,
    apply_mask,
    create_block_mask,
    create_feature_mask,
    create_random_mask,
    create_ssl_mask,
    create_timestep_mask,
)


class TestRandomMask:
    """Tests for random masking strategy."""

    def test_random_mask_shape(self):
        """Random mask should have the correct shape."""
        shape = (4, 48, 35)  # B, T, D
        mask = create_random_mask(shape, mask_ratio=0.15, device=torch.device("cpu"))

        assert mask.shape == shape
        assert mask.dtype == torch.bool

    def test_random_mask_ratio_approximate(self):
        """Random mask should approximately match target mask ratio."""
        shape = (10, 100, 50)  # Large enough for statistical test
        mask_ratio = 0.25
        mask = create_random_mask(shape, mask_ratio, device=torch.device("cpu"))

        # Count unmasked (True) values
        unmasked_ratio = mask.float().mean().item()
        expected_unmasked = 1.0 - mask_ratio

        # Allow 5% tolerance for randomness
        assert abs(unmasked_ratio - expected_unmasked) < 0.05

    def test_random_mask_reproducibility(self):
        """Random mask should be reproducible with generator."""
        shape = (4, 48, 35)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)

        mask1 = create_random_mask(shape, 0.15, torch.device("cpu"), gen1)
        mask2 = create_random_mask(shape, 0.15, torch.device("cpu"), gen2)

        assert torch.equal(mask1, mask2)


class TestBlockMask:
    """Tests for block masking strategy."""

    def test_block_mask_shape(self):
        """Block mask should have the correct shape."""
        shape = (4, 48, 35)
        mask = create_block_mask(shape, mask_ratio=0.3, device=torch.device("cpu"))

        assert mask.shape == shape
        assert mask.dtype == torch.bool

    def test_block_mask_creates_contiguous_blocks(self):
        """Block mask should create contiguous blocks of masked timesteps."""
        shape = (1, 48, 10)
        mask = create_block_mask(
            shape,
            mask_ratio=0.3,
            min_block_size=3,
            max_block_size=5,
            device=torch.device("cpu"),
        )

        # Check that masked regions are contiguous in time
        # For each batch, masked timesteps should form blocks
        time_mask = ~mask[0, :, 0]  # Get mask for first feature

        # Find transitions (0->1 or 1->0)
        transitions = (time_mask[1:] != time_mask[:-1]).sum().item()

        # With blocks, transitions should be limited (each block has 2 transitions at most)
        # If mask_ratio=0.3, we expect ~1-3 blocks, so 2-6 transitions
        assert transitions <= 10

    def test_block_mask_masks_all_features(self):
        """Block mask should mask entire timesteps (all features at once)."""
        shape = (1, 48, 10)
        mask = create_block_mask(shape, mask_ratio=0.3, device=torch.device("cpu"))

        # For each timestep, all features should have the same mask value
        for t in range(48):
            values_at_t = mask[0, t, :]
            assert torch.all(values_at_t == values_at_t[0])

    def test_block_mask_respects_ratio_approximately(self):
        """Block mask should approximately match target mask ratio."""
        shape = (10, 100, 35)
        mask_ratio = 0.4
        mask = create_block_mask(shape, mask_ratio, device=torch.device("cpu"))

        # Check mask ratio is within bounds
        actual_masked_ratio = (~mask).float().mean().item()
        # Block masking may slightly overshoot due to block constraints
        assert abs(actual_masked_ratio - mask_ratio) < 0.15


class TestTimestepMask:
    """Tests for timestep-wise masking strategy."""

    def test_timestep_mask_shape(self):
        """Timestep mask should have the correct shape."""
        shape = (4, 48, 35)
        mask = create_timestep_mask(shape, mask_ratio=0.2, device=torch.device("cpu"))

        assert mask.shape == shape
        assert mask.dtype == torch.bool

    def test_timestep_mask_entire_timesteps(self):
        """Timestep mask should mask entire timesteps across all features."""
        shape = (2, 48, 20)
        mask = create_timestep_mask(shape, mask_ratio=0.3, device=torch.device("cpu"))

        # For each batch and timestep, all features should have same mask
        for b in range(2):
            for t in range(48):
                values = mask[b, t, :]
                assert torch.all(values == values[0])

    def test_timestep_mask_independent_batches(self):
        """Timestep mask should be independent across batch dimension."""
        shape = (10, 48, 20)
        mask = create_timestep_mask(shape, mask_ratio=0.5, device=torch.device("cpu"))

        # Different batches should have different masks (with high probability)
        masks_per_batch = [mask[b, :, 0] for b in range(10)]
        unique_patterns = len(set(tuple(m.tolist()) for m in masks_per_batch))

        # With 10 batches and 50% masking, we expect most to be different
        assert unique_patterns >= 5


class TestFeatureMask:
    """Tests for feature-wise masking strategy."""

    def test_feature_mask_shape(self):
        """Feature mask should have the correct shape."""
        shape = (4, 48, 35)
        mask = create_feature_mask(shape, mask_ratio=0.2, device=torch.device("cpu"))

        assert mask.shape == shape
        assert mask.dtype == torch.bool

    def test_feature_mask_entire_features(self):
        """Feature mask should mask entire features across all timesteps."""
        shape = (2, 48, 20)
        mask = create_feature_mask(shape, mask_ratio=0.3, device=torch.device("cpu"))

        # For each batch and feature, all timesteps should have same mask
        for b in range(2):
            for d in range(20):
                values = mask[b, :, d]
                assert torch.all(values == values[0])


class TestApplyMask:
    """Tests for applying masks to tensors."""

    def test_apply_mask_zeros_masked(self):
        """apply_mask should set masked positions to mask_value."""
        x = torch.ones(2, 4, 3)
        mask = torch.zeros(2, 4, 3, dtype=torch.bool)  # All False = all masked

        result = apply_mask(x, mask, mask_value=0.0)

        assert torch.all(result == 0.0)

    def test_apply_mask_preserves_unmasked(self):
        """apply_mask should preserve values at unmasked positions."""
        x = torch.randn(2, 4, 3)
        mask = torch.ones(2, 4, 3, dtype=torch.bool)  # All True = none masked

        result = apply_mask(x, mask, mask_value=0.0)

        assert torch.equal(result, x)

    def test_apply_mask_partial(self):
        """apply_mask should handle partial masking correctly."""
        x = torch.ones(1, 4, 2) * 5.0
        mask = torch.tensor([[[True, False], [False, True], [True, True], [False, False]]])

        result = apply_mask(x, mask, mask_value=-1.0)

        expected = torch.tensor([[[5.0, -1.0], [-1.0, 5.0], [5.0, 5.0], [-1.0, -1.0]]])
        assert torch.equal(result, expected)


class TestCreateSSLMask:
    """Tests for the unified SSL mask creation function."""

    def test_ssl_mask_random_strategy(self):
        """SSL mask with 'random' strategy should use random masking."""
        shape = (4, 48, 35)
        mask = create_ssl_mask(shape, 0.15, strategy="random", device=torch.device("cpu"))

        assert mask.shape == shape
        assert mask.dtype == torch.bool

    def test_ssl_mask_block_strategy(self):
        """SSL mask with 'block' strategy should use block masking."""
        shape = (4, 48, 35)
        mask = create_ssl_mask(shape, 0.3, strategy="block", device=torch.device("cpu"))

        # Verify it has block structure
        time_mask = ~mask[0, :, 0]
        transitions = (time_mask[1:] != time_mask[:-1]).sum().item()
        assert transitions <= 20  # Block mask should have few transitions

    def test_ssl_mask_timestep_strategy(self):
        """SSL mask with 'timestep' strategy should mask entire timesteps."""
        shape = (2, 48, 20)
        mask = create_ssl_mask(shape, 0.3, strategy="timestep", device=torch.device("cpu"))

        for b in range(2):
            for t in range(48):
                values = mask[b, t, :]
                assert torch.all(values == values[0])

    def test_ssl_mask_feature_strategy(self):
        """SSL mask with 'feature' strategy should mask entire features."""
        shape = (2, 48, 20)
        mask = create_ssl_mask(shape, 0.3, strategy="feature", device=torch.device("cpu"))

        for b in range(2):
            for d in range(20):
                values = mask[b, :, d]
                assert torch.all(values == values[0])

    def test_ssl_mask_unknown_strategy_raises(self):
        """SSL mask with unknown strategy should raise ValueError."""
        shape = (4, 48, 35)
        with pytest.raises(ValueError, match="Unknown masking strategy"):
            create_ssl_mask(shape, 0.15, strategy="invalid", device=torch.device("cpu"))

    def test_ssl_mask_respects_obs_mask(self):
        """SSL mask should only mask observed values when obs_mask provided."""
        shape = (1, 10, 5)
        # Create obs_mask with some missing values
        obs_mask = torch.ones(shape, dtype=torch.bool)
        obs_mask[0, :3, :] = False  # First 3 timesteps are missing

        ssl_mask = create_ssl_mask(
            shape,
            mask_ratio=0.5,
            strategy="random",
            obs_mask=obs_mask,
            device=torch.device("cpu"),
        )

        # Missing values (where obs_mask=False) should NOT be masked for SSL
        # i.e., ssl_mask should be True where obs_mask is False
        assert torch.all(ssl_mask[0, :3, :] == True)  # noqa: E712


class TestBlockMaskOverlapCounting:
    """Test that block masking accurately counts masked positions with overlaps."""

    def test_mask_ratio_more_accurate_with_overlap_fix(self):
        """Actual mask ratio should be close to target despite overlapping blocks."""
        B, T, D = 100, 20, 5
        shape = (B, T, D)
        mask_ratio = 0.5
        generator = torch.Generator().manual_seed(42)

        mask = create_block_mask(
            shape,
            mask_ratio=mask_ratio,
            min_block_size=2,
            max_block_size=5,
            generator=generator,
        )

        masked_fraction = 1.0 - mask.float().mean().item()

        assert masked_fraction >= mask_ratio * 0.8, (
            f"Actual masked fraction {masked_fraction:.3f} is too far below "
            f"target {mask_ratio:.3f}"
        )

    def test_no_double_counting_in_small_sequence(self):
        """Verify mask count is correct even when blocks overlap."""
        B, T, D = 50, 20, 3
        shape = (B, T, D)
        mask_ratio = 0.5
        generator = torch.Generator().manual_seed(42)

        mask = create_block_mask(
            shape,
            mask_ratio=mask_ratio,
            min_block_size=3,
            max_block_size=8,
            generator=generator,
        )

        actual_masked = 1.0 - mask.float().mean().item()
        assert actual_masked >= mask_ratio * 0.85, (
            f"Average masked fraction {actual_masked:.3f} is too far below "
            f"target {mask_ratio:.3f}"
        )


class TestGaussianNoise:
    """Tests for Gaussian noise augmentation."""

    def test_gaussian_noise_changes_values(self):
        """Gaussian noise should modify input values."""
        x = torch.ones(4, 48, 35)
        noisy = apply_gaussian_noise(x, std=0.1)

        # Values should have changed
        assert not torch.equal(noisy, x)

    def test_gaussian_noise_mean_preserved(self):
        """Gaussian noise should approximately preserve mean (zero-mean noise)."""
        x = torch.ones(100, 48, 35) * 5.0
        noisy = apply_gaussian_noise(x, std=0.1)

        original_mean = x.mean().item()
        noisy_mean = noisy.mean().item()

        # Mean should be approximately preserved (within noise bounds)
        assert abs(noisy_mean - original_mean) < 0.05

    def test_gaussian_noise_respects_obs_mask(self):
        """Gaussian noise should only be added to observed values."""
        x = torch.ones(1, 10, 5) * 5.0
        obs_mask = torch.zeros(1, 10, 5, dtype=torch.bool)
        obs_mask[0, 5:, :] = True  # Only last 5 timesteps observed

        noisy = apply_gaussian_noise(x, std=0.5, obs_mask=obs_mask)

        # Missing values should be unchanged
        assert torch.all(noisy[0, :5, :] == 5.0)

        # Observed values should have noise added
        assert not torch.all(noisy[0, 5:, :] == 5.0)

    def test_gaussian_noise_std_parameter(self):
        """Gaussian noise std parameter should control noise magnitude."""
        x = torch.zeros(100, 48, 35)

        noisy_small = apply_gaussian_noise(x, std=0.01)
        noisy_large = apply_gaussian_noise(x, std=1.0)

        # Larger std should produce larger deviations
        assert noisy_large.std() > noisy_small.std()
