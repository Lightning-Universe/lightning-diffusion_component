from lightning.app import LightningWork


class LambdaWork(LightningWork):
    def __init__(self, fn, **kwargs):
        super().__init__(**kwargs)
        self.fn = fn
        self.model_store = "lit://model_store"

    def run(self):
        self.fn(self.model_store)
