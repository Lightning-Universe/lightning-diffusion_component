import lightning as L
import torch

class Work(L.LightningWork):

    def run(self):
        torch.zeros(1, device="cuda")

app = L.LightningApp(Work())