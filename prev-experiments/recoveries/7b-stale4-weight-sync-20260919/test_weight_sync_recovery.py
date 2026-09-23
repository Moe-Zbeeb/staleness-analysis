import ast
import asyncio
import contextlib
import hashlib
import io
import json
import os
import time
import types
import unittest
from pathlib import Path

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

import weight_sync_recovery as recovery


CAPTURE = Path(__file__).resolve().parents[3] / 'tmp/math-sweep-15b-normal-20260919/check_node10_weight_sync.json'
WATCHER_SHA256 = 'b9fe1f45122eb5dd64a109196dd1a78c6ec3ce684deaa28bd212d3b14f6db350'
TRANSPORT_SHA256 = 'e96806c5ddd3bf577818d311d7dd944e724829a84b676ccbc494e987efe7083d'


def original_sources():
    source_root = os.environ.get('WEIGHT_SYNC_ORIGINAL_SOURCE_ROOT')
    if source_root:
        root = Path(source_root)
        sources = {name: (root / name).read_text() for name in ['orchestrator/clients.py', 'orchestrator/watcher.py', 'transports/weights/nccl.py']}
    else:
        capture = json.loads(CAPTURE.read_text())
        sources = {name: record['text'] for name, record in capture['sources'].items()}
    expected = {
        'orchestrator/clients.py': recovery.ORIGINAL_CLIENTS_SHA256,
        'orchestrator/watcher.py': WATCHER_SHA256,
        'transports/weights/nccl.py': TRANSPORT_SHA256,
    }
    for name, digest in expected.items():
        if hashlib.sha256(sources[name].encode()).hexdigest() != digest:
            raise ValueError(f'Original source hash mismatch: {name}')
    return sources


class Logger:
    def debug(self, value):
        pass

    def warning(self, value):
        pass


def execute_selected(source, names, namespace):
    tree = ast.parse(source)
    selected = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
            selected.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module == 'weight_sync_recovery':
            selected.append(node)
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    exec(compile(module, 'captured_deployed_source.py', 'exec'), namespace)
    return namespace


def clients_namespace(source):
    namespace = {
        'asyncio': asyncio,
        'httpx': httpx,
        'AsyncRetrying': AsyncRetrying,
        'retry_if_exception': retry_if_exception,
        'stop_after_attempt': stop_after_attempt,
        'stop_after_delay': stop_after_delay,
        'wait_exponential': wait_exponential,
        'get_logger': Logger,
    }
    names = {'ADMIN_TIMEOUT_S', 'UPDATE_WEIGHTS_TIMEOUT_S', '_is_retryable_admin_error', '_admin_post', '_pause_engines', '_resume_engines', 'update_weights'}
    return execute_selected(source, names, namespace)


class Client:
    def __init__(self, events, *, fail_path=None, failures=1, fail_status=None, hold_update=None, update_entered=None):
        self.events = events
        self.fail_path = fail_path
        self.failures = failures
        self.fail_status = fail_status
        self.hold_update = hold_update
        self.update_entered = update_entered
        self.calls = []

    async def post(self, path, **kwargs):
        self.events.append(path)
        self.calls.append((path, kwargs))
        request = httpx.Request('POST', 'http://localhost:8574' + path)
        if path == self.fail_path and self.failures:
            self.failures -= 1
            if self.fail_status:
                return httpx.Response(self.fail_status, request=request)
            raise httpx.ReadTimeout('The server accepted this operation but its response was delayed', request=request)
        if path == '/update_weights' and self.hold_update is not None:
            self.update_entered.set()
            await self.hold_update.wait()
        return httpx.Response(200, request=request)


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = original_sources()
        cls.source = cls.sources['orchestrator/clients.py']
        cls.patched = recovery.apply_source_patch(cls.source)
        cls.clients = clients_namespace(cls.patched)
        receiver_namespace = execute_selected(
            cls.sources['transports/weights/nccl.py'],
            {'NCCLWeightReceiver'},
            {'WeightReceiver': object, 'update_weights': cls.clients['update_weights']},
        )
        cls.receiver_type = receiver_namespace['NCCLWeightReceiver']
        watcher_namespace = execute_selected(
            cls.sources['orchestrator/watcher.py'],
            {'WeightWatcher'},
            {'asyncio': asyncio, 'time': time, 'get_logger': Logger, 'format_time': str},
        )
        cls.watcher_type = watcher_namespace['WeightWatcher']

    def setUp(self):
        self.events = []
        self.output = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def receiver(self, client):
        receiver = self.receiver_type()
        receiver.admin_clients = [client]
        receiver.step_dir = lambda step: Path(f'/immutable/run/broadcast/step_{step}')
        receiver._ack = lambda step: self.events.append(f'ack:{step}')

        async def wait_published(step, cancelled=None):
            self.events.append(f'published:{step}')

        receiver.wait_published = wait_published
        return receiver

    def watcher(self, client):
        policy = types.SimpleNamespace(version=225)
        watcher = self.watcher_type(self.receiver(client), policy=policy, observers=[], ckpt_step=225)
        return watcher, policy

    def test_only_original_update_function_is_replaced(self):
        self.assertIs(self.clients['update_weights'], recovery.update_weights)
        original = ast.parse(self.source)
        patched = ast.parse(self.patched)
        original_nodes = [ast.dump(node, include_attributes=False) for node in original.body if not (isinstance(node, ast.AsyncFunctionDef) and node.name == 'update_weights')]
        patched_nodes = [ast.dump(node, include_attributes=False) for node in patched.body if not (isinstance(node, ast.ImportFrom) and node.module == 'weight_sync_recovery')]
        self.assertEqual(original_nodes, patched_nodes)

    def test_source_mismatch_and_double_patch_fail_closed(self):
        with self.assertRaises(ValueError):
            recovery.apply_source_patch(self.source + '\n')
        with self.assertRaises(ValueError):
            recovery.apply_source_patch(self.patched)

    async def test_actual_receiver_acknowledges_once_and_preserves_request(self):
        client = Client(self.events)
        await self.receiver(client).receive(226)
        self.assertEqual(self.events, ['/pause', 'ack:226', '/update_weights', '/resume'])
        self.assertEqual(client.calls[0][1]['params'], {'mode': 'keep', 'clear_cache': 'false'})
        self.assertEqual(client.calls[1][1]['json'], {'weight_dir': '/immutable/run/broadcast/step_226'})
        for _, kwargs in client.calls:
            self.assertEqual(kwargs['timeout'].as_dict(), {'connect': 30.0, 'read': 3600.0, 'write': 120.0, 'pool': 30.0})

    async def test_actual_watcher_advances_policy_only_after_confirmed_resume(self):
        client = Client(self.events)
        watcher, policy = self.watcher(client)
        await watcher.apply_policy_update(226)
        self.assertEqual(policy.version, 226)
        self.assertEqual(watcher.update_count, 1)
        self.assertEqual(self.events, ['published:226', '/pause', 'ack:226', '/update_weights', '/resume'])

    async def test_uncertain_update_timeout_never_retries_or_resumes(self):
        client = Client(self.events, fail_path='/update_weights')
        watcher, policy = self.watcher(client)
        with self.assertRaises(httpx.ReadTimeout):
            await watcher.apply_policy_update(226)
        self.assertEqual(self.events, ['published:226', '/pause', 'ack:226', '/update_weights'])
        self.assertEqual(policy.version, 225)
        self.assertEqual(watcher.update_count, 0)
        records = [json.loads(line) for line in self.output.getvalue().splitlines()]
        self.assertEqual(records[-1]['phase'], '/update_weights')
        self.assertEqual(records[-1]['error_type'], 'ReadTimeout')

    async def test_update_http_500_never_retries_or_resumes(self):
        client = Client(self.events, fail_path='/update_weights', fail_status=500)
        with self.assertRaises(httpx.HTTPStatusError):
            await self.receiver(client).receive(226)
        self.assertEqual(self.events, ['/pause', 'ack:226', '/update_weights'])

    async def test_pause_timeout_never_acknowledges_or_updates(self):
        client = Client(self.events, fail_path='/pause')
        with self.assertRaises(httpx.ReadTimeout):
            await self.receiver(client).receive(226)
        self.assertEqual(self.events, ['/pause'])

    async def test_resume_timeout_does_not_advance_policy_or_replay_update(self):
        client = Client(self.events, fail_path='/resume')
        watcher, policy = self.watcher(client)
        with self.assertRaises(httpx.ReadTimeout):
            await watcher.apply_policy_update(226)
        self.assertEqual(self.events, ['published:226', '/pause', 'ack:226', '/update_weights', '/resume'])
        self.assertEqual(policy.version, 225)
        self.assertEqual(watcher.update_count, 0)

    async def test_delayed_accepted_update_stays_single_and_blocks_policy(self):
        release = asyncio.Event()
        entered = asyncio.Event()
        client = Client(self.events, hold_update=release, update_entered=entered)
        watcher, policy = self.watcher(client)
        task = asyncio.create_task(watcher.apply_policy_update(226))
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0)
            self.assertEqual(self.events.count('/update_weights'), 1)
            self.assertNotIn('/resume', self.events)
            self.assertEqual(policy.version, 225)
            release.set()
            await asyncio.wait_for(task, 1)
            self.assertEqual(self.events.count('/update_weights'), 1)
            self.assertEqual(policy.version, 226)
        finally:
            if not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def test_cancelled_inflight_update_is_not_retried_or_resumed(self):
        release = asyncio.Event()
        entered = asyncio.Event()
        client = Client(self.events, hold_update=release, update_entered=entered)
        watcher, policy = self.watcher(client)
        task = asyncio.create_task(watcher.apply_policy_update(226))
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(self.events, ['published:226', '/pause', 'ack:226', '/update_weights'])
        self.assertEqual(policy.version, 225)

    async def test_original_deployed_code_replays_an_uncertain_update(self):
        original = clients_namespace(self.source)
        client = Client(self.events, fail_path='/update_weights')
        await original['update_weights']([client], Path('/immutable/run/broadcast/step_226'), step=226, on_paused=lambda: self.events.append('ack:226'))
        self.assertEqual(self.events, ['/pause', 'ack:226', '/update_weights', '/update_weights', '/resume'])

    async def test_wrong_inference_topology_is_rejected_before_any_post(self):
        client = Client(self.events)
        with self.assertRaises(ValueError):
            await recovery.update_weights([client, client], Path('/immutable/run/broadcast/step_226'), step=226)
        self.assertEqual(self.events, [])


if __name__ == '__main__':
    unittest.main()
