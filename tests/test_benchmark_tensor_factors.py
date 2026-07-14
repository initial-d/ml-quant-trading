from scripts.benchmark_tensor_factors import _benchmark_row, _environment_row


def test_benchmark_environment_row_escapes_pipes():
    row = _environment_row("Platform", "Linux | CI")

    assert "Linux \\| CI" in row


def test_benchmark_result_row_escapes_pipes():
    row = _benchmark_row("cpu|0", "ts_mean(close|adj,20)", 0.001, 0.0002, "1 | MB")

    assert "cpu\\|0" in row
    assert "ts_mean(close\\|adj,20)" in row
    assert "1 \\| MB" in row
