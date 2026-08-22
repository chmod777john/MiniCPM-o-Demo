"""CPU-only tests for the modeling-layer strategy-HD scheduler."""

from MiniCPMO45.modeling_minicpmo import MiniCPMODuplex


def _scheduler(*, enabled=True, hd_slices=4):
    duplex = object.__new__(MiniCPMODuplex)
    duplex.strategy_hd = enabled
    duplex.strategy_hd_max_slice_nums = hd_slices
    duplex._reset_streaming_state()
    return duplex


def test_strategy_hd_is_lag_one_and_resets():
    duplex = _scheduler()

    assert duplex._resolve_max_slice_nums(None) == 1
    duplex._update_strategy_after_generate(is_listen=False)
    assert duplex._resolve_max_slice_nums(None) == 4
    duplex._update_strategy_after_generate(is_listen=False)
    assert duplex._resolve_max_slice_nums(None) == 4
    duplex._update_strategy_after_generate(is_listen=True)
    assert duplex._resolve_max_slice_nums(None) == 1

    duplex._update_strategy_after_generate(is_listen=False)
    duplex._reset_streaming_state()
    assert duplex._resolve_max_slice_nums(None) == 1


def test_strategy_hd_owns_explicit_slice_value():
    duplex = _scheduler()
    duplex._update_strategy_after_generate(is_listen=False)

    assert duplex._resolve_max_slice_nums(2) == 4
    assert duplex._resolve_max_slice_nums(None) == 4


def test_strategy_disabled_keeps_explicit_slice_value():
    duplex = _scheduler(enabled=False)
    duplex._update_strategy_after_generate(is_listen=False)
    assert duplex._resolve_max_slice_nums(None) == 1
    assert duplex._resolve_max_slice_nums(2) == 2
