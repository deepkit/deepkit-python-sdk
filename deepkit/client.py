import asyncio
import json
import os
import sys
import threading
import time
from typing import Dict, List

import websockets
from rx.subject import BehaviorSubject

import deepkit.globals
import inspect


def is_in_directory(filepath, directory):
    return os.path.realpath(filepath).startswith(os.path.realpath(directory))


class Client(threading.Thread):
    connection: websockets.WebSocketClientProtocol

    def __init__(self, config_path: str):
        self.connected = BehaviorSubject(False)
        self.config_path = config_path
        self.loop = asyncio.new_event_loop()
        self.host = os.environ.get('DEEPKIT_HOST', '127.0.0.1')
        self.port = int(os.environ.get('DEEPKIT_PORT', '8960'))
        self.token = os.environ.get('DEEPKIT_JOB_ACCESSTOKEN', None)
        self.job_id = os.environ.get('DEEPKIT_JOB_ID', None)
        self.message_id = 0
        self.account = 'localhost'
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

    def run(self):
        self.connecting = self.loop.create_future()
        self.loop.create_task(self._connect())
        self.loop.run_forever()

    def shutdown(self):
        if self.offline: return
        promise = asyncio.run_coroutine_threadsafe(self.stop_and_sync(), self.loop)
        promise.result()
        if not self.connection.closed:
            raise Exception('Connection still active')
        self.loop.stop()

    async def stop_and_sync(self):
        self.stopping = True

        # todo, assign correct status depending on python exit code

        # done = 150, //when all tasks are done
        # aborted = 200, //when at least one task aborted
        # failed = 250, //when at least one task failed
        # crashed = 300, //when at least one task crashed
        self.patches['status'] = 150
        self.patches['ended'] = time.time() * 1000

        self.patches['tasks.main.ended'] = time.time() * 1000

        # done = 500,
        # aborted = 550,
        # failed = 600,
        # crashed = 650,
        self.patches['tasks.main.status'] = 500

        self.patches['tasks.main.instances.0.ended'] = time.time() * 1000
        # done = 500,
        # aborted = 550,
        # failed = 600,
        # crashed = 650,
        self.patches['tasks.main.instances.0.status'] = 500

        while len(self.patches) > 0 or len(self.queue) > 0:
            await asyncio.sleep(0.15)

        await self.connection.close()

    async def register_controller(self, name: str, controller):
        self.controllers[name] = controller

        async def subscriber(message, done):
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
                        'replyId': message['replyId'],
                        'data': {'type': 'error', 'id': 0, 'stack': None, 'entityName': '@error:default',
                                 'error': error}
                    }, no_response=True)

                if data['name'] == 'actionTypes':
                    parameters = []
                    for arg in inspect.getfullargspec(getattr(controller, data['action'])).args:
                        parameters.append({
                            'type': 'Any',
                            'array': False,
                            'partial': False
                        })

                    await self._message({
                        'name': 'peerController/message',
                        'controllerName': name,
                        'replyId': message['replyId'],
                        'data': {
                            'type': 'actionTypes/result',
                            'id': 0,
                            'parameters': parameters,
                            'returnType': {'partial': False, 'type': 'Any', 'array': False}
                        }
                    }, no_response=True)

                if data['name'] == 'action':
                    try:
                        res = getattr(controller, data['action'])(*data['args'])

                        await self._message({
                            'name': 'peerController/message',
                            'controllerName': name,
                            'replyId': message['replyId'],
                            'data': {
                                'type': 'next/json',
                                'id': message['id'],
                                'next': res,
                            }
                        }, no_response=True)
                    except Exception as e:
                        await self._message({
                            'name': 'peerController/message',
                            'controllerName': name,
                            'replyId': message['replyId'],
                            'data': {'type': 'error', 'id': 0, 'stack': None, 'entityName': '@error:default',
                                     'error': str(e)}
                        }, no_response=True)

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
                })

        return Controller(self)

    async def _action(self, controller: str, action: str, args: List, lock=True):
        if lock: await self.connecting
        if self.offline: return
        if self.stopping: raise Exception('In shutdown: actions disallowed')
        res = await self._message({
            'name': 'action',
            'controller': controller,
            'action': action,
            'args': args,
            'timeout': 60
        }, lock=lock)
        if res['type'] == 'next/json':
            return res['next']

        if res['type'] == 'error':
            print(res, file=sys.stderr)
            raise Exception('API Error: ' + res['error'])

        raise Exception(f"Invalid action type '{res['type']}'. Not implemented")

    def job_action(self, action: str, args: List):
        return asyncio.run_coroutine_threadsafe(self._action('job', action, args), self.loop)

    async def _subscribe(self, message, subscriber):
        await self.connecting

        self.message_id += 1
        message['id'] = self.message_id

        message_id = self.message_id

        def on_done():
            del self.subscriber[message_id]

        async def on_incoming_message(incoming_message):
            await subscriber(incoming_message, on_done)

        self.subscriber[self.message_id] = on_incoming_message
        self.queue.append(message)

    async def _message(self, message, lock=True, no_response=False):
        if lock: await self.connecting

        self.message_id += 1
        message['id'] = self.message_id
        if not no_response:
            self.callbacks[self.message_id] = self.loop.create_future()

        self.queue.append(message)

        if no_response:
            return

        return await self.callbacks[self.message_id]

    def patch(self, path: str, value: any):
        if self.offline: return
        if self.stopping: raise Exception('In shutdown: patches disallowed')

        self.patches[path] = value

    async def send_messages(self, connection):
        while not connection.closed:
            try:
                q = self.queue[:]
                for m in q:
                    await connection.send(json.dumps(m))
                    self.queue.remove(m)
            except Exception:
                return

            if len(self.patches) > 0:
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
                    }))

                    for i in send.keys():
                        if self.patches[i] == send[i]:
                            del self.patches[i]

                except websockets.exceptions.ConnectionClosed:
                    return

            await asyncio.sleep(0.25)

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
                    await self.subscriber[res['id']](res)

                if res['id'] in self.callbacks:
                    self.callbacks[res['id']].set_result(res)
                    del self.callbacks[res['id']]

        if not self.stopping:
            print("Deepkit: lost connection. reconnect ...")
            self.connecting = self.loop.create_future()
            self.connected.on_next(False)
            self.loop.create_task(self._connect())

    async def _connect_job(self, host: str, port: int, id: str, token: str):
        try:
            self.connection = await websockets.connect(f"ws://{host}:{port}")
        except Exception:
            # try again later
            await asyncio.sleep(1)
            self.loop.create_task(self._connect())
            return

        self.loop.create_task(self.handle_messages(self.connection))
        self.loop.create_task(self.send_messages(self.connection))

        res = await self._message({
            'name': 'authenticate',
            'token': {
                'id': 'job',
                'token': token,
                'job': id
            }
        }, lock=False)

        if not res['result'] or res['result'] is not True:
            raise Exception('Job token invalid')

        # load job controller
        await self._message({
            'name': 'action',
            'controller': 'job',
            'action': ''
        }, lock=False)

        self.connecting.set_result(True)
        if self.connections > 0:
            print("Deepkit: Reconnected.")

        self.connected.on_next(True)
        self.connections += 1

    async def _connect(self):
        # we want to restart with a empty queue, so authentication happens always first
        queue_copy = self.queue[:]
        self.queue = []

        if self.token:
            await self._connect_job(self.host, self.port, self.job_id, self.token)
        else:
            account_config = self.get_account_config(self.account)
            self.host = account_config['host']
            self.port = account_config['port']

            try:
                self.connection = await websockets.connect(f"ws://{self.host}:{self.port}")
            except Exception as e:
                self.offline = True
                print(f"Deepkit: App not started or server not reachable. Monitoring disabled. {e}")
                self.connecting.set_result(False)
                return

            self.loop.create_task(self.handle_messages(self.connection))
            self.loop.create_task(self.send_messages(self.connection))
            res = await self._message({
                'name': 'authenticate',
                'token': {
                    'id': 'user',
                    'token': account_config['token']
                }
            }, lock=False)
            if not res['result']:
                raise Exception('Login invalid')

            link = self.get_folder_link()

            deepkit_config_yaml = None
            if self.config_path and os.path.exists(self.config_path):
                deepkit_config_yaml = open(self.config_path, 'r', encoding='utf-8').read()

            job = await self._action('app', 'createJob', [link['projectId'], self.config_path, deepkit_config_yaml],
                                     lock=False)
            deepkit.globals.loaded_job = job
            self.token = await self._action('app', 'getJobAccessToken', [job['id']], lock=False)
            self.job_id = job['id']
            await self.connection.close()
            await self._connect_job(self.host, self.port, self.job_id, self.token)

        self.queue = queue_copy + self.queue

    def get_account_config(self, name: str) -> Dict:
        with open(os.path.expanduser('~') + '/.deepkit/config', 'r') as h:
            config = json.load(h)
            for account in config['accounts']:
                if account['name'] == name:
                    return account

        raise Exception(f"No account for {name} found.")

    def get_folder_link(self) -> Dict:
        with open(os.path.expanduser('~') + '/.deepkit/config', 'r') as h:
            config = json.load(h)
            for link in config['folderLinks']:
                if is_in_directory(sys.path[0], link['path']):
                    return link

        raise Exception(f"No project link for {sys.path[0]} found.")
