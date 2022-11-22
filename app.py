# ! sudo apt update
# ! curl -fsSL https://code-server.dev/install.sh | sh
import lightning as L
from vscode3 import VSCode

app = L.LightningApp(
    VSCode("dreambooth_component.py"),
    flow_cloud_compute=L.CloudCompute("gpu-fast"),
)
