"""Verify ``lumilake jobs submit --help`` documents the ``--input`` syntax."""

from lumilake_cli.commands.job import app
from typer.testing import CliRunner


def test_submit_help_shows_input_syntax_examples() -> None:
    result = CliRunner().invoke(app, ["submit", "--help"])
    assert result.exit_code == 0
    output = result.stdout
    assert "Name=val1,val2,val3" in output
    assert "--input query=NVDA,TSLA" in output
    assert "Name=path.txt" in output
