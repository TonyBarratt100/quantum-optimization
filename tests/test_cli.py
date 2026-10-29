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


def test_demo_runs_on_a_fake_ibm_device_without_asking(capsys):
    assert (
        main(
            ["--backend", "fake_marrakesh", "--layers", "1", "--restarts", "2", "--shots", "500", "--no-plot"]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Backend: fake_marrakesh (local fake, real device noise)" in output
    assert "P(max) corrected" in output


def test_real_hardware_asks_first_and_sends_nothing_if_declined(monkeypatch, capsys):
    from qaoa_maxcut import hardware

    fake = hardware.load_backend("fake_marrakesh")
    monkeypatch.setattr(hardware, "load_backend", lambda name: fake)  # no real account used
    monkeypatch.setattr(hardware, "run_on_backend", lambda *a, **k: pytest.fail("must not run"))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert main(["--backend", "ibm_marrakesh", "--restarts", "2", "--no-plot"]) == 1
    output = capsys.readouterr().out
    assert "5 circuits x 4,000 shots" in output
    assert "Cancelled: nothing was sent to IBM." in output


@pytest.mark.parametrize("arguments", [["--layers", "6"], ["--shots", "50"]])
def test_hardware_limits_are_enforced(arguments):
    with pytest.raises(SystemExit):
        main(["--backend", "fake_marrakesh", *arguments, "--no-plot"])
