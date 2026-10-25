import numpy as np

from simulate import block_bootstrap_indices


def test_block_bootstrap_shape_and_range():
    positions = block_bootstrap_indices(500, 20, 252, 10, np.random.default_rng(42))
    assert positions.shape == (20, 252)
    assert positions.min() >= 0
    assert positions.max() <= 499


def test_block_bootstrap_shape_when_horizon_is_not_a_multiple_of_block():
    positions = block_bootstrap_indices(500, 7, 25, 10, np.random.default_rng(42))
    assert positions.shape == (7, 25)


def test_block_bootstrap_is_reproducible_for_a_fixed_seed():
    first = block_bootstrap_indices(500, 20, 252, 10, np.random.default_rng(42))
    second = block_bootstrap_indices(500, 20, 252, 10, np.random.default_rng(42))
    other = block_bootstrap_indices(500, 20, 252, 10, np.random.default_rng(43))
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_block_bootstrap_blocks_are_consecutive_and_wrap_around():
    n_obs = 50
    positions = block_bootstrap_indices(n_obs, 200, 100, 10, np.random.default_rng(0))
    steps = (positions[:, 1:] - positions[:, :-1]) % n_obs
    inside_block = np.arange(99) % 10 != 9  # the step between two blocks is not consecutive
    assert (steps[:, inside_block] == 1).all()
