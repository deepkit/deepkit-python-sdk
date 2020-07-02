import asyncio
import inspect
import json
import os
import sys
import threading
from asyncio import Future
import datetime
from enum import Enum
from typing import Dict, Optional

import numpy as np
import websockets
from rx.subject import BehaviorSubject

import deepkit.globals
from deepkit.home import get_home_config
from deepkit.model import FolderLink


def is_in_directory(filepath, directory):
    return os.path.realpath(filepath).startswith(os.path.realpath(directory))


class ApiError(Exception):
    pass


def json_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.float):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime.datetime):
        # we assume all datetime instances are UTC
        return obj.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    else:
        return str(obj)


class JobStatus(Enum):
    done = 150  # when all tasks are done
    aborted = 200  # when at least one task aborted
    failed = 250  # when at least one task failed
    crashed = 300  # when at least one task crashed


class Client(threading.Thread):
    connection: websockets.WebSocketClientProtocol

    def __init__(self, project: Optional[str] = None,
                 account: Optional[str] = None,
                 try_pick_up=False,
                 parent_experiment=None,
                 silent=False):
        self.connected = BehaviorSubject(False)
        self.project = project
        self.account = account
        self.parent_experiment = parent_experiment
        self.silent = silent

        self.host = os.environ.get('DEEPKIT_HOST', '127.0.0.1')
        self.socket_path = os.environ.get('DEEPKIT_SOCKET', None)
        self.ssl = os.environ.get('DEEPKIT_SSL', '0') is '1'
        self.port = int(os.environ.get('DEEPKIT_PORT', '8960'))

        self.job_token = None
        self.job_id = None

        if try_pick_up:
            # is set by Deepkit cli
            self.job_token = os.environ.get('DEEPKIT_JOB_ACCESSTOKEN', None)
            self.job_id = os.environ.get('DEEPKIT_JOB_ID', None)

        # is set by deepkit.login()
        self.token = os.environ.get('DEEPKIT_ACCESSTOKEN', None)

        self.result_status = None

        self.message_id = 0
        self.callbacks: Dict[int, asyncio.Future] = {}
        self.subscriber: Dict[int, any] = {}
        self.stopping = False
        self.queue = []
        self.controllers = {}
        self.patches = {}
        self.offline = False
        self.connections = 0
        self.lock = threading.Lock()
        threading.Thread.__init__(self)
        self.daemon = True
        self.loop = asyncio.new_event_loop()
        self.start()

    def is_connected(self):
        return self.connected.value

    def run(self):
        self.connecting = self.loop.create_future()
        self.loop.run_forever()

    def connect(self):
        asyncio.run_coroutine_threadsafe(self._connect(), self.loop)

    def connect_anon(self):
        asyncio.run_coroutine_threadsafe(self._connect_anon(), self.loop).result()

    def shutdown(self):
        if self.offline: return
        promise = asyncio.run_coroutine_threadsafe(self.stop_and_sync(), self.loop)
        promise.result()
        if not self.connection.closed:
            raise Exception('Connection still active')
        self.loop.stop()

    async def stop_and_sync(self):
        self.stopping = True

        if deepkit.utils.in_self_execution() or self.result_status:
            # only when we are in self execution do we set status, time stamps etc
            # otherwise the CLI is doing that and the server. Or when
            # the experiment set result_state explicitly.

            # done = 150, //when all tasks are done
            # aborted = 200, //when at least one task aborted
            # failed = 250, //when at least one task failed
            # crashed = 300, //when at least one task crashed
            self.patches['status'] = 150
            self.patches['ended'] = datetime.datetime.utcnow()
            self.patches['tasks.main.ended'] = datetime.datetime.utcnow()

            # done = 500,
            # aborted = 550,
            # failed = 600,
            # crashed = 650,
            self.patches['tasks.main.status'] = 500
            self.patches['tasks.main.instances.0.ended'] = datetime.datetime.utcnow()

            # done = 500,
            # aborted = 550,
            # failed = 600,
            # crashed = 650,
            self.patches['tasks.main.instances.0.status'] = 500

            if hasattr(sys, 'last_value'):
                if isinstance(sys.last_value, KeyboardInterrupt):
                    self.patches['status'] = 200
                    self.patches['tasks.main.status'] = 550
                    self.patches['tasks.main.instances.0.status'] = 550
                else:
                    self.patches['status'] = 300
                    self.patches['tasks.main.status'] = 650
                    self.patches['tasks.main.instances.0.status'] = 650

            if self.result_status:
                self.patches['status'] = self.result_status.value

        while len(self.patches) > 0 or len(self.queue) > 0:
            await asyncio.sleep(0.15)

        await self.connection.close()

    def register_controller(self, name: str, controller):
        return asyncio.run_coroutine_threadsafe(self._register_controller(name, controller), self.loop)

    async def _register_controller(self, name: str, controller):
        self.controllers[name] = controller

        async def handle_peer_message(message, done):
            if message['type'] == 'error':
                done()
                del self.controllers[name]
                raise Exception('Register controller error: ' + message['error'])

            if message['type'] == 'ack':
                pass

            if message['type'] == 'peerController/message':
                data = message['data']

                if not hasattr(controller, data['action']):
                    error = f"Requested action {message['action']} not available in {name}"
                    print(error, file=sys.stderr)
                    await self._message({
                        'name': 'peerController/message',
                        'controllerName': name,
                        'clientId': message['clientId'],
                        'data': {'type': 'error', 'id': data['id'], 'stack': None, 'entityName': '@error:default',
                                 'error': error}
                    }, no_response=True)

                if data['name'] == 'actionTypes':
                    parameters = []

                    i = 0
                    for arg in inspect.getfullargspec(getattr(controller, data['action'])).args:
                        parameters.append({
                            'type': 'any',
                            'name': '#' + str(i)
                        })
                        i += 1

                    await self._message({
                        'name': 'peerController/message',
                        'controllerName': name,
                        'clientId': message['clientId'],
                        'data': {
                            'type': 'actionTypes/result',
                            'id': data['id'],
                            'parameters': parameters,
                            'returnType': {'type': 'any', 'name': 'result'}
                        }
                    }, no_response=True)

                if data['name'] == 'action':
                    try:
                        res = await getattr(controller, data['action'])(*data['args'])

                        await self._message({
                            'name': 'peerController/message',
                            'controllerName': name,
                            'clientId': message['clientId'],
                            'data': {
                                'type': 'next/json',
                                'id': data['id'],
                                'encoding': {'name': 'r', 'type': 'any'},
                                'next': res,
                            }
                        }, no_response=True)
                    except Exception as e:
                        await self._message({
                            'name': 'peerController/message',
                            'controllerName': name,
                            'clientId': message['clientId'],
                            'data': {'type': 'error', 'id': data['id'], 'stack': None, 'entityName': '@error:default',
                                     'error': str(e)}
                        }, no_response=True)

        def subscriber(message, on_done):
            self.loop.create_task(handle_peer_message(message, on_done))

        await self._subscribe({
            'name': 'peerController/register',
            'controllerName': name,
        }, subscriber)

        class Controller:
            def __init__(self, client):
                self.client = client

            def stop(self):
                self.client._message({
                    'name': 'peerController/unregister',
                    'controllerName': name,
                }, no_response=True)

        return Controller(self)

    async def _action(self, controller: str, action: str, args=None, lock=True, allow_in_shutdown=False):
        if args is None:
            args = []

        if lock: await self.connecting
        if self.offline: return
        if self.stopping and not allow_in_shutdown: raise Exception('In shutdown: actions disallowed')

        if not controller: raise Exception('No controller given')
        if not action: raise Exception('No action given')

        # print('> action', action, threading.current_thread().name)
        res = await self._message({
            'name': 'action',
            'controller': controller,
            'action': action,
            'args': args,
            'timeout': 60
        }, lock=lock)

        # print('< action', action)

        if res['type'] == 'next/json':
            return res['next'] if 'next' in res else None

        if res['type'] == 'error':
            print(res, file=sys.stderr)
            raise ApiError('API Error: ' + str(res['error']))

        raise ApiError(f"Invalid action type '{res['type']}'. Not implemented")

    def app_action_threadsafe(self, action: str, args=None) -> Future:
        if args is None: args = []
        return asyncio.run_coroutine_threadsafe(self._action('app', action, args), self.loop)

    async def job_action(self, action: str, args=None):
        return await self._action('job', action, args)

    def job_action_threadsafe(self, action: str, args=None) -> Future:
        """
        This method is non-blocking and every try to block-wait for an answers means
        script execution stops when connection is broken (offline training entirely impossible).
        So, we just schedule the call and return a Future, which the user can subscribe to.
        """
        if args is None: args = []
        return asyncio.run_coroutine_threadsafe(self._action('job', action, args), self.loop)

    async def _subscribe(self, message, subscriber):
        await self.connecting

        self.message_id += 1
        message['id'] = self.message_id

        message_id = self.message_id

        def on_done():
            del self.subscriber[message_id]

        def on_incoming_message(incoming_message):
            subscriber(incoming_message, on_done)

        self.subscriber[self.message_id] = on_incoming_message
        self.queue.append(message)

    def _create_message(self, message: dict, lock=True, no_response=False) -> dict:
        self.message_id += 1
        message['id'] = self.message_id
        if not no_response:
            self.callbacks[self.message_id] = self.loop.create_future()

        return message

    async def _message(self, message, lock=True, no_response=False):
        if lock: await self.connecting

        message = self._create_message(message, no_response=no_response)
        self.queue.append(message)

        if no_response:
            return

        return await self.callbacks[self.message_id]

    def patch(self, path: str, value: any):
        if self.offline: return
        if self.stopping: return

        self.patches[path] = value

    async def send_messages(self, connection):
        while not connection.closed:
            try:
                q = self.queue[:]
                for m in q:
                    try:
                        j = json.dumps(m, default=json_converter)
                    except TypeError as e:
                        print('Could not send message since JSON error', e, m, file=sys.stderr)
                        continue
                    await connection.send(j)
                    self.queue.remove(m)
            except Exception as e:
                print("Failed sending, exit send_messages", file=sys.stderr)
                raise e

            if len(self.patches) > 0:
                # we have to send first all messages/actions out
                # before sending patches, as most of the time
                # patches are based on previously created entities,
                # so we need to make sure those entities are created
                # first before sending any patches.
                # print('patches', self.patches)
                try:
                    send = self.patches.copy()
                    await connection.send(json.dumps({
                        'name': 'action',
                        'controller': 'job',
                        'action': 'patchJob',
                        'args': [
                            send
                        ],
                        'timeout': 60
                    }, default=json_converter))

                    for i in send.keys():
                        if self.patches[i] == send[i]:
                            del self.patches[i]
                except websockets.exceptions.ConnectionClosed:
                    return
                except ApiError:
                    print("Patching failed. Syncing job data disabled.", file=sys.stderr)
                    return

            await asyncio.sleep(0.5)

    async def handle_messages(self, connection):
        while not connection.closed:
            try:
                res = json.loads(await connection.recv())
            except websockets.exceptions.ConnectionClosedError:
                # we need reconnect
                break
            except websockets.exceptions.ConnectionClosedOK:
                # we closed on purpose, so no reconnect necessary
                return

            if res and 'id' in res:
                if res['id'] in self.subscriber:
                    self.subscriber[res['id']](res)

                if res['id'] in self.callbacks:
                    self.callbacks[res['id']].set_result(res)
                    del self.callbacks[res['id']]

        if not self.stopping:
            self.log("Deepkit: lost connection. reconnect ...")
            self.connecting = self.loop.create_future()
            self.connected.on_next(False)
            self.loop.create_task(self._connect())

    async def _connected(self, id: str, token: str):
        try:
            if self.socket_path:
                self.connection = await websockets.unix_connect(self.socket_path)
            else:
                ws = 'wss' if self.ssl else 'ws'
                url = f"{ws}://{self.host}:{self.port}"
                self.connection = await websockets.connect(url)
        except Exception as e:
            # try again later
            self.log('Unable to connect', e)
            await asyncio.sleep(1)
            self.loop.create_task(self._connect())
            return

        self.loop.create_task(self.handle_messages(self.connection))
        # we don't use send_messages() since this would send all queue/patches
        # which would lead to permission issues when we're not first authenticated

        if token:
            message = self._create_message({
                'name': 'authenticate',
                'token': {
                    'id': 'job',
                    'token': token,
                    'job': id
                }
            }, lock=False)

            await self.connection.send(json.dumps(message, default=json_converter))

            res = await self.callbacks[message['id']]
            if not res['result'] or res['result'] is not True:
                raise Exception('Job token invalid')

        self.loop.create_task(self.send_messages(self.connection))

        self.connecting.set_result(True)
        if self.connections > 0:
            self.log("Deepkit: Reconnected.")

        self.connected.on_next(True)
        self.connections += 1

    async def _connect_anon(self):
        ws = 'wss' if self.ssl else 'ws'
        url = f"{ws}://{self.host}:{self.port}"
        self.connection = await websockets.connect(url)
        self.loop.create_task(self.handle_messages(self.connection))
        self.loop.create_task(self.send_messages(self.connection))

        self.connecting.set_result(True)
        self.connected.on_next(True)
        self.connections += 1

    async def _connect(self):
        # we want to restart with a empty queue, so authentication happens always first
        queue_copy = self.queue[:]
        self.queue = []

        if self.job_token:
            await self._connected(self.job_id, self.job_token)
            return

        try:
            link: Optional[FolderLink] = None

            user_token = self.token
            account_name = 'none'

            if not user_token:
                config = get_home_config()
                # when no user_token is given (via deepkit.login() for example)
                # we need to find the host, port, token from the user config in ~/.deepkit/config
                if not self.account and not self.project:
                    # find both, start with
                    link = config.get_folder_link_of_directory(sys.path[0])
                    account_config = config.get_account_for_id(link.accountId)
                elif self.account and not self.project:
                    account_config = config.get_account_for_name(self.account)
                else:
                    # default to first account configured
                    account_config = config.get_first_account()

                account_name = account_config.name
                self.host = account_config.host
                self.port = account_config.port
                self.ssl = account_config.ssl
                user_token = account_config.token

            ws = 'wss' if self.ssl else 'ws'
            try:
                url = f"{ws}://{self.host}:{self.port}"
                self.connection = await websockets.connect(url)
            except Exception as e:
                self.offline = True
                print(f"Deepkit: App not started or server not reachable. Monitoring disabled. {e}", file=sys.stderr)
                self.connecting.set_result(False)
                return

            self.loop.create_task(self.handle_messages(self.connection))
            self.loop.create_task(self.send_messages(self.connection))

            res = await self._message({
                'name': 'authenticate',
                'token': {
                    'id': 'user',
                    'token': user_token
                }
            }, lock=False)
            if not res['result']:
                raise Exception('Login invalid')

            project_name = ''
            if link:
                project_name = link.name
                projectId = link.projectId
            else:
                if not self.project:
                    raise Exception('No project defined. Please use project="project-name" '
                                    'to specify which project to use.')

                project = await self._action('app', 'getProjectForPublicName', [self.project], lock=False)

                if not project:
                    raise Exception(
                        f'No project found for name {self.project}. Make sure it exists before using it. '
                        f'Do you use the correct account? (used {account_name})')
                project_name = project['name']
                projectId = project['id']

            job = await self._action('app', 'createJob', [projectId, self.parent_experiment],
                                     lock=False)

            prefix = "Sub experiment" if self.parent_experiment else "Experiment"
            self.log(f"{prefix} #{job['number']} created in project {project_name} using account {account_name}")

            deepkit.globals.loaded_job_config = job['config']['config']
            self.job_token = await self._action('app', 'getJobAccessToken', [job['id']], lock=False)
            self.job_id = job['id']

            # todo, implement re-authentication, so we don't have to drop the active connection
            await self.connection.close()
            await self._connected(self.job_id, self.job_token)
        except Exception as e:
            self.connecting.set_exception(e)

        self.queue = queue_copy + self.queue

    def log(self, *message: str):
        if not self.silent: print(*message)
