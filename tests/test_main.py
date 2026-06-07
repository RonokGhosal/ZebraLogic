"""Tests for zebralogic.main."""

from zebralogic.main import greet, main


def test_greet_default():
    assert greet().startswith("Hello, world!")


def test_greet_custom_name():
    assert "Ronok" in greet("Ronok")


def test_main_returns_zero(capsys):
    exit_code = main(["--name", "tester"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "tester" in captured.out
