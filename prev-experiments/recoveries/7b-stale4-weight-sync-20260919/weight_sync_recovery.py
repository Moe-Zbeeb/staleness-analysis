from __future__ import annotations

import ast
import datetime
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import httpx
from httpx import AsyncClient


ORIGINAL_CLIENTS_SHA256 = 'a2d117b78b34c638cc19f3a92e8ec166634a23120319a14ae51ea570afa9fcc7'
READ_TIMEOUT_SECONDS = 3600.0
CONNECT_TIMEOUT_SECONDS = 30.0
WRITE_TIMEOUT_SECONDS = 120.0
POOL_TIMEOUT_SECONDS = 30.0


def apply_source_patch(source: str) -> str:
    if hashlib.sha256(source.encode()).hexdigest() != ORIGINAL_CLIENTS_SHA256:
        raise ValueError('The original clients.py does not match the captured deployed source')
    tree = ast.parse(source)
    targets = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == 'update_weights']
    if len(targets) != 1 or targets[0].decorator_list:
        raise ValueError('Expected exactly one undecorated update_weights function')
    target = targets[0]
    lines = source.splitlines(keepends=True)
    replacement = 'from weight_sync_recovery import update_weights as update_weights\n'
    patched = ''.join(lines[:target.lineno - 1]) + replacement + ''.join(lines[target.end_lineno:])
    compile(patched, 'recovery/orchestrator/clients.py', 'exec')
    return patched


def emit_event(step: int, phase: str, status: str, elapsed: float, error: BaseException | None = None) -> None:
    record = {
        'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'event': 'weight_sync_recovery',
        'step': step,
        'phase': phase,
        'status': status,
        'attempt': 1,
        'elapsed_seconds': round(elapsed, 6),
        'read_timeout_seconds': READ_TIMEOUT_SECONDS,
    }
    if error is not None:
        record['error_type'] = type(error).__name__
    print(json.dumps(record, sort_keys=True), flush=True)


async def post_once(client: AsyncClient, path: str, *, step: int, **kwargs) -> None:
    started = time.monotonic()
    emit_event(step, path, 'started', 0.0)
    try:
        response = await client.post(
            path,
            timeout=httpx.Timeout(
                connect=CONNECT_TIMEOUT_SECONDS,
                read=READ_TIMEOUT_SECONDS,
                write=WRITE_TIMEOUT_SECONDS,
                pool=POOL_TIMEOUT_SECONDS,
            ),
            **kwargs,
        )
        response.raise_for_status()
    except BaseException as error:
        emit_event(step, path, 'failed', time.monotonic() - started, error)
        raise
    emit_event(step, path, 'succeeded', time.monotonic() - started)


async def update_weights(
    admin_clients: list[AsyncClient],
    weight_dir: Path | None,
    step: int = 0,
    on_paused: Callable[[], None] | None = None,
) -> None:
    if len(admin_clients) != 1:
        raise ValueError('This recovery is restricted to one inference admin client')
    weight_dir_posix = weight_dir.as_posix() if weight_dir is not None else None
    client = admin_clients[0]
    await post_once(client, '/pause', step=step, params={'mode': 'keep', 'clear_cache': 'false'})
    if on_paused is not None:
        on_paused()
    await post_once(client, '/update_weights', step=step, json={'weight_dir': weight_dir_posix})
    await post_once(client, '/resume', step=step)
