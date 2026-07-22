from pathlib import Path

from plantxai_stability.cli import main


def test_smoke_cli_has_twelve_scenarios(capsys) -> None:
    code = main(["smoke", str(Path("configs/protocol/v0.9/protocol.yaml"))])
    assert code == 0
    output = capsys.readouterr().out
    assert '"scenario_count": 12' in output
