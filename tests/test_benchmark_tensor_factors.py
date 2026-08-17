import pytest

from scripts.benchmark_tensor_factors import (
    PROTOCOL_VERSION,
    _benchmark_row,
    _configure_threads,
    _environment_row,
)


def test_benchmark_environment_row_escapes_pipes():
    row = _environment_row("Platform", "Linux | CI")

    assert "Linux \\| CI" in row


def test_benchmark_result_row_escapes_pipes():
    row = _benchmark_row("cpu|0", "ts_mean(close|adj,20)", 0.001, 0.0002, "1 | MB")

    assert "cpu\\|0" in row
    assert "ts_mean(close\\|adj,20)" in row
    assert "1 \\| MB" in row


def test_benchmark_protocol_version_is_explicit():
    assert PROTOCOL_VERSION == "v1"


def test_configure_threads_pins_both_pools(monkeypatch):
    calls = []
    monkeypatch.setattr("torch.set_num_threads", lambda value: calls.append(("intra", value)))
    monkeypatch.setattr("torch.set_num_interop_threads", lambda value: calls.append(("interop", value)))

    _configure_threads(4, 1)

    assert calls == [("intra", 4), ("interop", 1)]


@pytest.mark.parametrize("threads,interop_threads", [(0, 1), (1, 0), (-1, 1)])
def test_configure_threads_rejects_non_positive_counts(threads, interop_threads):
    with pytest.raises(ValueError, match="positive"):
        _configure_threads(threads, interop_threads)
