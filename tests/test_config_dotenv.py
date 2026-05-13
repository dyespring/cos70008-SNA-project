"""Tests for the .env loader's override behaviour.

Sensitive keys (LLM provider config, API tokens) need to be **swappable
by editing .env alone** — a stale ``export DASHSCOPE_API_KEY=…`` in the
user's shell rc shouldn't silently mask a fresh value in .env. The
loader uses a small allow-list (``_DOTENV_OVERRIDE_KEYS``) for this; an
escape hatch (``DOTENV_RESPECT_SHELL=1``) restores the old behaviour.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture
def env_sandbox(monkeypatch, tmp_path):
    """Snapshot env vars and route _load_dotenv at a temp .env file."""
    from src import config as config_mod

    # Anything we touch is restored automatically by monkeypatch on teardown.
    return {"path": tmp_path / ".env", "module": config_mod, "mp": monkeypatch}


def _write_env(path: Path, **kv: str) -> None:
    path.write_text(
        "\n".join(f"{k}={v}" for k, v in kv.items()) + "\n",
        encoding="utf-8",
    )


def test_non_sensitive_key_respects_existing_env(env_sandbox, monkeypatch):
    """For non-allow-listed keys, shell wins over .env (legacy behaviour)."""
    _write_env(env_sandbox["path"], MY_REGULAR_KEY="from-dotenv")
    monkeypatch.setenv("MY_REGULAR_KEY", "from-shell")

    env_sandbox["module"]._load_dotenv(env_sandbox["path"])

    assert os.environ["MY_REGULAR_KEY"] == "from-shell"


def test_sensitive_key_overrides_existing_env(env_sandbox, monkeypatch):
    """For keys in _DOTENV_OVERRIDE_KEYS, .env wins over a stale shell export."""
    _write_env(env_sandbox["path"], DASHSCOPE_API_KEY="sk-from-dotenv")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-stale-shell")

    env_sandbox["module"]._load_dotenv(env_sandbox["path"])

    assert os.environ["DASHSCOPE_API_KEY"] == "sk-from-dotenv"


def test_dotenv_respect_shell_disables_override(env_sandbox, monkeypatch):
    """The escape hatch restores the legacy 'shell wins' behaviour."""
    _write_env(env_sandbox["path"], DASHSCOPE_API_KEY="sk-from-dotenv")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-stale-shell")
    monkeypatch.setenv("DOTENV_RESPECT_SHELL", "1")

    env_sandbox["module"]._load_dotenv(env_sandbox["path"])

    assert os.environ["DASHSCOPE_API_KEY"] == "sk-stale-shell"


def test_missing_env_key_filled_in(env_sandbox, monkeypatch):
    """If the env var doesn't exist yet, .env value is taken regardless."""
    _write_env(env_sandbox["path"], NEW_VAR="new-value")
    monkeypatch.delenv("NEW_VAR", raising=False)

    env_sandbox["module"]._load_dotenv(env_sandbox["path"])

    assert os.environ["NEW_VAR"] == "new-value"


def test_missing_dotenv_is_silent(env_sandbox):
    """No .env file → no exception, no change."""
    nonexistent = env_sandbox["path"].with_name("does_not_exist.env")
    env_sandbox["module"]._load_dotenv(nonexistent)


def test_quoted_values_are_unquoted(env_sandbox, monkeypatch):
    """``KEY="value"`` and ``KEY='value'`` should both yield ``value``."""
    env_sandbox["path"].write_text(
        'DBL_QUOTED="hello"\nSGL_QUOTED=\'world\'\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("DBL_QUOTED", raising=False)
    monkeypatch.delenv("SGL_QUOTED", raising=False)

    env_sandbox["module"]._load_dotenv(env_sandbox["path"])

    assert os.environ["DBL_QUOTED"] == "hello"
    assert os.environ["SGL_QUOTED"] == "world"


def test_comments_and_blanks_ignored(env_sandbox):
    env_sandbox["path"].write_text(
        "# this is a comment\n\n   \nFOO=bar\n",
        encoding="utf-8",
    )
    env_sandbox["module"]._load_dotenv(env_sandbox["path"])
    assert os.environ.get("FOO") == "bar"


def test_override_keys_includes_critical_credentials(env_sandbox):
    """Lock the allow-list contract — anyone widening it is welcome,
    but accidentally removing a credential key would be a regression."""
    keys = env_sandbox["module"]._DOTENV_OVERRIDE_KEYS
    for required in (
        "LLM_PROVIDER", "LLM_MODEL",
        "OPENAI_API_KEY", "DASHSCOPE_API_KEY",
        "HUGGINGFACE_API_TOKEN",
    ):
        assert required in keys, f"{required} should be in _DOTENV_OVERRIDE_KEYS"
