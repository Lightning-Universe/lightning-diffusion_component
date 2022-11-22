import subprocess
from lightning import LightningFlow, LightningWork
from lightning.app.utilities.load_app import load_app_from_file
import traceback
from lightning.app.utilities.app_helpers import _LightningAppRef
from lightning.app.utilities.enum import CacheCallsKeys
from lightning.app.utilities.exceptions import CacheMissException


class VSCodeServer(LightningWork):
    def __init__(self):
        super().__init__(parallel=True, start_with_flow=False)
        self._process = None

    def run(self):
        self._process = subprocess.Popen(f"code-server --auth=none . --bind-addr={self.host}:{self.port}", shell=True)

    def on_exit(self):
        self._process.kill()

class VSCode(LightningFlow):

    def __init__(self, entrypoint_file: str):
        super().__init__()
        self.entrypoint_file = entrypoint_file
        self.flow = None
        self.vscode = VSCodeServer()
        self.should_reload = True

    def run(self):

        self.vscode.run()

        if self.should_reload:

            if self.flow:
                for w in self.flow.works():
                    latest_hash = w._calls[CacheCallsKeys.LATEST_CALL_HASH]
                    if latest_hash is not None:
                        w.delete()

            try:
                # Loading another Lightning App Reference.
                app = _LightningAppRef().get_current()
                new_flow = load_app_from_file(self.entrypoint_file).root
                _LightningAppRef._app_instance = app
                new_flow = self.upgrade_fn(self.flow, new_flow)
                del self.flow
                self.flow = new_flow
                self._start_with_flow_works(self.flow)
            except Exception:
                print(traceback.print_exc())

            self.should_reload = False

        try:
            if self.flow:
                self.flow.run()
        except CacheMissException as e:
            raise e
        except Exception:
            print(traceback.print_exc())

    def configure_layout(self):
        tabs = [{"name": "vscode", "content": self.vscode}]
        if self.flow:
            try:
                new_tabs = self.flow.configure_layout()
                # TODO: Validate new_tabs format.
                tabs += new_tabs
            except Exception:
                print(traceback.print_exc())
        return tabs

    def upgrade_fn(self, old_flow, new_flow):
        return new_flow

    def reload(self):
        self.should_reload = True

    def configure_commands(self):
        return [{"reload": self.reload}]

    @staticmethod
    def _start_with_flow_works(flow):
        for w in flow.works():
            if w._start_with_flow:
                parallel = w.parallel
                w._parallel = True
                w.start()
                w._parallel = parallel