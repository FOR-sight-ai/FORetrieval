"""Shared SSH connection helper.

Opens a paramiko ``SSHClient`` while honouring ``~/.ssh/config`` so that
host aliases, ``HostName``, ``User``, ``Port``, ``IdentityFile``,
``ProxyCommand`` and ``ProxyJump`` all behave the same way as the ``ssh``
command-line tool.

This is the single source of truth for SSH connections used by every
deployment manager (currently the embedding server and the vector-DB
server).  Keeping it in one place avoids drift between managers.

The helper is deliberately minimal — it returns a connected
``SSHClient`` and lets callers run commands or open SFTP sessions
themselves.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _resolve_identity_files(host_cfg: Dict[str, Any]) -> Optional[Union[str, List[str]]]:
    """Extract ``IdentityFile`` entries from a paramiko SSHConfig lookup.

    paramiko returns this as a list when set, even when there is only one
    entry.  We pass the value through unchanged to paramiko.connect which
    accepts either a single path or a list of paths.
    """
    identity = host_cfg.get("identityfile")
    if not identity:
        return None
    # paramiko.SSHConfig already expands ~ for IdentityFile values.
    return identity


def _build_proxy_command(host_cfg: Dict[str, Any]) -> Optional[str]:
    """Return a ProxyCommand string from a paramiko SSHConfig lookup.

    Explicit ``ProxyCommand`` wins.  If absent, ``ProxyJump`` is
    translated to an equivalent ``ssh -W %h:%p <jump>`` invocation.
    Returns ``None`` when neither is configured.
    """
    pc = host_cfg.get("proxycommand")
    if pc:
        return pc
    pj = host_cfg.get("proxyjump")
    if pj:
        # Translate ProxyJump to a ProxyCommand for paramiko's sock arg.
        return f"ssh -W %h:%p {pj}"
    return None


def open_ssh_client(
    ssh_host: str,
    ssh_user: Optional[str] = None,
    ssh_key_path: Optional[str] = None,
    ssh_config_path: Optional[Path] = None,
):
    """Open a connected paramiko SSHClient using ``~/.ssh/config`` semantics.

    Args:
        ssh_host: The (possibly aliased) host name.  This is the same
            string you would pass to the ``ssh`` command.
        ssh_user: Optional override for the SSH user.  When set, it wins
            over any ``User`` directive in ``~/.ssh/config``.
        ssh_key_path: Optional override for the identity file.  When
            set, it wins over any ``IdentityFile`` directive.
        ssh_config_path: Optional path to an SSH config file.  Defaults
            to ``~/.ssh/config``.  Pass a custom path in tests.

    Returns:
        A connected ``paramiko.SSHClient``.  Caller is responsible for
        closing it.

    Raises:
        ImportError: paramiko is not installed.
        paramiko.SSHException: connection failed.
        socket.gaierror: DNS resolution failed for the resolved host.
        OSError: low-level network error (refused, unreachable, …).
    """
    import paramiko

    if ssh_config_path is None:
        ssh_config_path = Path("~/.ssh/config").expanduser()

    # Load and look up the host alias, if any.
    host_cfg: Dict[str, Any] = {}
    if ssh_config_path.exists():
        try:
            cfg = paramiko.SSHConfig()
            with ssh_config_path.open() as f:
                cfg.parse(f)
            host_cfg = cfg.lookup(ssh_host)
            logger.debug(
                "SSH config lookup for '%s' -> %s",
                ssh_host,
                {k: v for k, v in host_cfg.items() if k != "identityfile"},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to parse %s (%s) — falling back to direct connect",
                ssh_config_path, exc,
            )
            host_cfg = {}

    # Resolve effective connect args.
    resolved_host: str = host_cfg.get("hostname", ssh_host)
    resolved_port: int = int(host_cfg.get("port", 22))
    resolved_user: Optional[str] = (
        ssh_user
        or host_cfg.get("user")
        or os.environ.get("USER")
        or "root"
    )

    connect_kwargs: Dict[str, Any] = {
        "hostname": resolved_host,
        "port": resolved_port,
        "username": resolved_user,
        "allow_agent": True,
        "look_for_keys": True,
    }

    # Identity file: explicit override > SSHConfig IdentityFile > paramiko default.
    if ssh_key_path:
        connect_kwargs["key_filename"] = ssh_key_path
    else:
        identity = _resolve_identity_files(host_cfg)
        if identity is not None:
            connect_kwargs["key_filename"] = identity

    # ProxyCommand / ProxyJump
    proxy_cmd = _build_proxy_command(host_cfg)
    if proxy_cmd:
        # Expand the standard tokens that paramiko's SSHConfig.lookup() may
        # leave in the string. We follow OpenSSH semantics: %h = remote
        # hostname, %p = remote port, %r = remote user.
        expanded = (
            proxy_cmd
            .replace("%h", resolved_host)
            .replace("%p", str(resolved_port))
            .replace("%r", resolved_user or "")
        )
        logger.debug("Using ProxyCommand: %s", expanded)
        connect_kwargs["sock"] = paramiko.ProxyCommand(expanded)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # Load known_hosts so AutoAddPolicy doesn't blow up the first time.
    try:
        client.load_system_host_keys()
    except Exception:  # noqa: BLE001
        pass

    client.connect(**connect_kwargs)
    return client
