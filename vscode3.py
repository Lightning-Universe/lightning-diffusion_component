import subprocess
import threading
from lightning import LightningFlow, LightningWork, BuildConfig
from lightning.app.utilities.load_app import load_app_from_file
from watchfiles import watch, PythonFilter
import traceback
from lightning.app.utilities.app_helpers import _LightningAppRef
from time import sleep

class PythonWatcher(threading.Thread):

    def __init__(self, component):
        super().__init__(daemon=True)
        self.component = component

    def run(self):
        try:
            self.component.should_reload = True

            while self.component.should_reload:
                sleep(1)

            for _ in watch('.', watch_filter=PythonFilter(ignore_paths=[__file__])):

                self.component.should_reload = True

                while self.component.should_reload:
                    sleep(1)

        except Exception as e:
            print(traceback.print_exc())


class VSCodeBuildConfig(BuildConfig):
    def build_commands(self):
        return [
            "sudo apt update",
            "curl -fsSL https://code-server.dev/install.sh | sh",
        ]


class VSCodeServer(LightningWork):
    def __init__(self):
        super().__init__(
            cloud_build_config=VSCodeBuildConfig(),
            parallel=True,
            start_with_flow=False,
        )
        self.should_reload = False
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
        self.should_reload = False
        self._thread = None

    def run(self):
        if self._thread is None:
            self._thread = PythonWatcher(self)
            self._thread.start()

        self.vscode.run()

        if self.should_reload:

            if self.flow:
                for w in self.flow.works():
                    w.stop()

            try:
                # Loading another Lightning App Reference.
                app = _LightningAppRef().get_current()
                new_flow = load_app_from_file(self.entrypoint_file).root
                _LightningAppRef._app_instance = app
                new_flow = self.upgrade_fn(self.flow, new_flow)
                del self.flow
                self.flow = new_flow
            except Exception:
                print(traceback.print_exc())

            self.should_reload = False

        if self.flow:
            self.flow.run()

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