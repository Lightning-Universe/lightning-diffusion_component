import subprocess
import threading
from lightning import LightningFlow, LightningWork, CloudCompute, BuildConfig
from lightning.app.utilities.load_app import load_app_from_file
from watchfiles import watch, PythonFilter
import traceback
import os

class PythonWatcher(threading.Thread):

    def __init__(self, work):
        super().__init__(daemon=True)
        self.work = work

    def run(self):
        try:
            self.work.should_reload = True

            for _ in watch('.', watch_filter=PythonFilter(ignore_paths=[__file__])):

                self.work.should_reload = False
                self.work.should_reload = True

        except Exception as e:
            print(traceback.print_exc())


class VSCodeBuildConfig(BuildConfig):
    def build_commands(self):
        return [
            "sudo apt update",
            "sudo apt install python3.8-venv",
            "curl -fsSL https://code-server.dev/install.sh | sh",
        ]


class VSCodeServer(LightningWork):
    def __init__(self):
        super().__init__(
            cloud_build_config=VSCodeBuildConfig(),
            parallel=True,
        )
        self.should_reload = False
        self._thread = None

    def run(self):
        self._thread = PythonWatcher(self)
        self._thread.start()
        # subprocess.call("mkdir ~/playground && cd ~/playground && python -m venv venv", shell=True)
        subprocess.call(f"code-server --auth=none . --bind-addr={self.host}:{self.port}", shell=True)

    def on_exit(self):
        self._thread.join(0)


class VSCode(LightningFlow):

    def __init__(self, entrypoint_file: str):
        super().__init__()
        os.environ["ENABLE_MULTIPLE_WORKS_IN_DEFAULT_CONTAINER"] = "1"

        self.entrypoint_file = entrypoint_file
        self.vscode = VSCodeServer()
        self.flow = None

    def run(self):
        self.vscode.run()

        if self.vscode.should_reload:

            if self.flow:
                for w in self.flow.works():
                    w.stop()

            try:
                new_flow = load_app_from_file(self.entrypoint_file).root
                new_flow = self.upgrade_fn(self.flow, new_flow)
                del self.flow
                self.flow = new_flow
            except Exception:
                print(traceback.print_exc())

            self.vscode.should_reload = False

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