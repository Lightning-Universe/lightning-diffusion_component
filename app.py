# ! sudo apt update
# ! curl -fsSL https://code-server.dev/install.sh | sh
from lightning import LightningApp
from vscode3 import VSCode

app = LightningApp(VSCode("component.py"))
