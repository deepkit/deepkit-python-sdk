import base64
import io
import time
from typing import NamedTuple, List, Dict

import PIL.Image


class DebuggerFetchItem(NamedTuple):
    name: str
    output: any
    ahistogram: any
    whistogram: any
    bhistogram: any


class DebuggerFetchConfig(NamedTuple):
    x: int
    layers: List[str]
    all: bool

    def needs_fetch(self, name: str) -> bool:
        if self.all: return True
        return name in self.layers


def pil_image_to_jpeg(image):
    buffer = io.BytesIO()

    image.save(buffer, format="PNG", quality=70)
    return buffer.getvalue()

class DebuggerManager:
    def __init__(self, context):
        import deepkit
        self.context: deepkit.Context = context

        self.live_last_sent = time.time()
        self.x = 0
        self.record_snapshot_created = False
        self.record_last_sent = time.time()
        self.record_last_epoch = 0
        self.debuggers = []
        self.active_debug_data_for_this_run = False
        self.record_needed = False
        self.live_needed = False
        self.send_data_futures = []

    def register_debugger(self, debugger):
        self.debuggers.append(debugger)

    def on_disconnect(self):
        for f in self.send_data_futures:
            f.set_result(False)

        self.send_data_futures = []

    def create_snapshot(self, x, layers):
        self.context.client.job_action_threadsafe('addSnapshot', [
            x,
            time.time(),
            layers,
            self.context.job_iteration,
            self.context.job_step,
        ])

    def tick(self):
        """
        Checks whether a new snapshot or live data needs to be fetched and sent. If so we trigger on each debugger
        instance a fetch() call and send that data to the server.
        """
        if self.active_debug_data_for_this_run: return
        if not self.context.client.is_connected(): return

        state = self.context.debugger_controller.state
        self.record_needed = state.recording
        fetch_all = False

        if state.recordingMode == 'second':
            diff = time.time() - self.record_last_sent
            if diff <= state.recordingSecond:
                # not enough time past, wait for next call
                self.record_needed = False

        if state.recordingMode == 'epoch':
            # if not epoch_end: record_needed = False
            if self.context.job_iteration == self.record_last_epoch:
                # nothing to do for records
                self.record_needed = False

        self.live_needed = state.live and (time.time() - self.live_last_sent) > 1
        layers = list(state.watchingLayers.keys())

        if not self.live_needed and not self.record_needed:
            return

        self.active_debug_data_for_this_run = True
        self.record_snapshot_created = False

        if self.record_needed and state.recordingLayers == 'all':
            fetch_all = True

        # wait for all previous to be sent first.
        try:
            for f in self.send_data_futures: f.result()
        except Exception as e:
            print('Failing sending debug data', e)
            pass

        self.x += 1

        fetch_config = DebuggerFetchConfig(x=self.x, layers=layers, all=fetch_all)

        fetch_layers: Dict[str, DebuggerFetchItem] = dict()
        for debugger in self.debuggers:
            fetch_layers.update(debugger.fetch(fetch_config))

        if self.record_needed and len(fetch_layers):
            self.create_snapshot(self.x, list(fetch_layers.keys()))

        for fetch in fetch_layers.values():
            output = fetch.output
            output_image = None
            if isinstance(fetch.output, PIL.Image.Image):
                output = None
                output_image = base64.b64encode(pil_image_to_jpeg(fetch.output)).decode()

            if self.record_needed:
                self.send_data_futures.append(self.context.client.job_action_threadsafe('setSnapshotLayerData', [
                    fetch_config.x,
                    self.live_needed,
                    fetch.name,
                    output,
                    output_image,
                    base64.b64encode(fetch.ahistogram).decode() if fetch.ahistogram else None,
                    base64.b64encode(fetch.whistogram).decode() if fetch.whistogram else None,
                    base64.b64encode(fetch.bhistogram).decode() if fetch.bhistogram else None,
                ]))
            else:
                self.send_data_futures.append(self.context.client.job_action_threadsafe('addLiveLayerData', [
                    fetch.name,
                    output,
                    output_image,
                    base64.b64encode(fetch.ahistogram).decode() if fetch.ahistogram else None,
                    base64.b64encode(fetch.whistogram).decode() if fetch.whistogram else None,
                    base64.b64encode(fetch.bhistogram).decode() if fetch.bhistogram else None,
                ]))

        self.live_last_sent = time.time()

        self.active_debug_data_for_this_run = False

        if self.record_needed:
            self.record_last_sent = time.time()

        if self.live_needed:
            self.live_last_sent = time.time()

        self.record_needed = False
        self.live_needed = False
        self.record_snapshot_created = False