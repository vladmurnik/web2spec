from __future__ import annotations

from web2spec import cli


def test_version() -> None:
    assert cli.main(["--version"]) == 0
    assert cli.VERSION == "0.4.0"


def test_help_lists_main_commands() -> None:
    text = cli._help()
    for command in ("collect", "search", "research", "mcp", "doctor", "eval-q4"):
        assert command in text
    assert "20-50" in text


def test_research_maps_to_research_script() -> None:
    assert cli.COMMANDS["research"] == "26_research.py"
    assert cli.COMMANDS["research"] != cli.COMMANDS["search"]


def test_unknown_command_returns_two(capsys) -> None:
    assert cli.main(["nope"]) == 2
    assert "unknown command" in capsys.readouterr().err


def test_all_mapped_scripts_exist() -> None:
    missing = [
        script for script in cli.COMMANDS.values() if not (cli.SCRIPT_DIR / script).is_file()
    ]
    assert missing == []
