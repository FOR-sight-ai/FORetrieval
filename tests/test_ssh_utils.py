"""Tests for foretrieval.ssh_utils.open_ssh_client.

All tests mock paramiko.SSHClient.connect so nothing touches the network.
The goal is to confirm that ~/.ssh/config directives are honoured the
same way as the OpenSSH CLI.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _write_ssh_config(tmp_path: Path, content: str) -> Path:
    """Write an SSH config to a temp file and return its path."""
    p = tmp_path / "ssh_config"
    p.write_text(content)
    return p


def _captured_connect_kwargs(mock_client_class):
    """Return the kwargs of the most recent SSHClient.connect call."""
    instance = mock_client_class.return_value
    return instance.connect.call_args.kwargs


# ---------------------------------------------------------------------------
# Basic resolution
# ---------------------------------------------------------------------------

class TestHostAliasResolution:
    def test_alias_resolves_to_hostname(self, tmp_path):
        cfg = _write_ssh_config(tmp_path, """
Host pf01
    HostName pf01.example.com
    User alice
    Port 2222
""")
        from foretrieval import ssh_utils
        with patch("paramiko.SSHClient") as MockClient:
            ssh_utils.open_ssh_client("pf01", ssh_config_path=cfg)
        kw = _captured_connect_kwargs(MockClient)
        assert kw["hostname"] == "pf01.example.com"
        assert kw["port"] == 2222
        assert kw["username"] == "alice"

    def test_no_config_falls_back_to_direct(self, tmp_path):
        """Missing config file → direct connect with the raw host string."""
        from foretrieval import ssh_utils
        with patch("paramiko.SSHClient") as MockClient:
            ssh_utils.open_ssh_client(
                "real.example.com", ssh_config_path=tmp_path / "missing"
            )
        kw = _captured_connect_kwargs(MockClient)
        assert kw["hostname"] == "real.example.com"
        assert kw["port"] == 22

    def test_no_match_falls_back_to_direct(self, tmp_path):
        cfg = _write_ssh_config(tmp_path, """
Host someotheralias
    HostName irrelevant
""")
        from foretrieval import ssh_utils
        with patch("paramiko.SSHClient") as MockClient:
            ssh_utils.open_ssh_client("pf01", ssh_config_path=cfg)
        kw = _captured_connect_kwargs(MockClient)
        assert kw["hostname"] == "pf01"


# ---------------------------------------------------------------------------
# Override precedence
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_user_override_beats_ssh_config(self, tmp_path):
        cfg = _write_ssh_config(tmp_path, """
Host pf01
    HostName pf01.example.com
    User alice
""")
        from foretrieval import ssh_utils
        with patch("paramiko.SSHClient") as MockClient:
            ssh_utils.open_ssh_client(
                "pf01", ssh_user="override_user", ssh_config_path=cfg,
            )
        kw = _captured_connect_kwargs(MockClient)
        assert kw["username"] == "override_user"

    def test_key_path_override_beats_identityfile(self, tmp_path):
        cfg = _write_ssh_config(tmp_path, """
Host pf01
    HostName pf01.example.com
    IdentityFile ~/.ssh/should_be_ignored
""")
        from foretrieval import ssh_utils
        with patch("paramiko.SSHClient") as MockClient:
            ssh_utils.open_ssh_client(
                "pf01",
                ssh_key_path="/explicit/key/path",
                ssh_config_path=cfg,
            )
        kw = _captured_connect_kwargs(MockClient)
        assert kw["key_filename"] == "/explicit/key/path"


# ---------------------------------------------------------------------------
# ProxyCommand / ProxyJump
# ---------------------------------------------------------------------------

class TestProxy:
    def test_proxycommand_creates_sock(self, tmp_path):
        cfg = _write_ssh_config(tmp_path, """
Host pf01
    HostName pf01.example.com
    User alice
    ProxyCommand ssh -W %h:%p bastion
""")
        from foretrieval import ssh_utils
        with patch("paramiko.SSHClient") as MockClient, \
             patch("paramiko.ProxyCommand") as MockPC:
            ssh_utils.open_ssh_client("pf01", ssh_config_path=cfg)
        # ProxyCommand instance was constructed and passed as 'sock'
        kw = _captured_connect_kwargs(MockClient)
        assert "sock" in kw
        # The expanded command was used
        MockPC.assert_called_once()
        expanded_cmd = MockPC.call_args.args[0]
        assert "pf01.example.com" in expanded_cmd
        assert ":22" in expanded_cmd
        assert "bastion" in expanded_cmd

    def test_proxyjump_translates_to_proxycommand(self, tmp_path):
        cfg = _write_ssh_config(tmp_path, """
Host pf01
    HostName pf01.example.com
    ProxyJump bastion.example.com
""")
        from foretrieval import ssh_utils
        with patch("paramiko.SSHClient") as MockClient, \
             patch("paramiko.ProxyCommand") as MockPC:
            ssh_utils.open_ssh_client("pf01", ssh_config_path=cfg)
        kw = _captured_connect_kwargs(MockClient)
        assert "sock" in kw
        MockPC.assert_called_once()
        cmd = MockPC.call_args.args[0]
        assert "ssh -W" in cmd
        assert "bastion.example.com" in cmd

    def test_no_proxy_no_sock(self, tmp_path):
        cfg = _write_ssh_config(tmp_path, """
Host pf01
    HostName pf01.example.com
""")
        from foretrieval import ssh_utils
        with patch("paramiko.SSHClient") as MockClient:
            ssh_utils.open_ssh_client("pf01", ssh_config_path=cfg)
        kw = _captured_connect_kwargs(MockClient)
        assert "sock" not in kw


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------

class TestFailureModes:
    def test_malformed_ssh_config_falls_back_to_direct(self, tmp_path):
        cfg = _write_ssh_config(tmp_path, "this is not valid ssh config syntax{{{")
        # paramiko.SSHConfig is permissive — write something that *does*
        # crash its parser. Easiest: simulate the parse exception.
        from foretrieval import ssh_utils
        with patch("paramiko.SSHConfig") as MockCfg, \
             patch("paramiko.SSHClient") as MockClient:
            MockCfg.return_value.parse.side_effect = RuntimeError("bad syntax")
            ssh_utils.open_ssh_client("pf01", ssh_config_path=cfg)
        kw = _captured_connect_kwargs(MockClient)
        assert kw["hostname"] == "pf01"
