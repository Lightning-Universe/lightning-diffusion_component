import lightning as L
from lightning_diffusion import BaseDiffusion
from lightning_diffusion.diffusion_serve import DreamBoothInput, DreamBoothOutput
from lightning.app.runners import MultiProcessRuntime
import torch
import requests

class ServeDiffusion(BaseDiffusion):

    def setup(self, *args, **kwargs):
        self._model = torch.ones(1).to(self.device)

    def predict(self, input = DreamBoothInput):
        tensor = torch.tensor(int(input.prompt)).to(self.device)
        out = self.model + tensor
        return DreamBoothOutput(image=str(out.cpu().item()))


class LightningAppTest(L.LightningApp):
    
    def run_once(self):
        res = super().run_once()
        server = list(self.root.load_balancer.works())[0]
        if server.url:
            resp = requests.post(f"{server.url}/predict", json=DreamBoothInput(prompt="2").dict())
            assert resp.status_code == 200, resp.json()
            return True
        return res

def test_base_diffusion(monkeypatch):

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    app = LightningAppTest(ServeDiffusion())
    MultiProcessRuntime(app).dispatch()