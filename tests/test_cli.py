import pytest

from qaoa_maxcut.__main__ import main


def test_demo_prints_a_table_and_saves_the_plot(tmp_path, capsys):
    plot_path = tmp_path / "out.png"
    assert main(["--layers", "1", "--restarts", "2", "--shots", "200", "--plot", str(plot_path)]) == 0
    output = capsys.readouterr().out
    assert "Maximum cut (brute force): 7" in output
    assert plot_path.stat().st_size > 0


@pytest.mark.parametrize("flag", ["--layers", "--restarts", "--shots"])
def test_demo_rejects_values_below_one(flag):
    with pytest.raises(SystemExit):
        main([flag, "0", "--no-plot"])
