# ! sudo apt update"
# ! curl -fsSL https://code-server.dev/install.sh | sh
import lightning as L
import time

class Work(L.LightningWork):

    def run(self):
        print("RUNNING ON GPU")

class Flow(L.LightningFlow):

    def __init__(self):
        super().__init__()
        self.counter = -10
        # self.work = Work(cloud_compute=L.CloudCompute("gpu"))

    def run(self):
        self.counter += 1
        print(self.counter)
        # self.work.run()
        time.sleep(1)

    def configure_layout(self):
        return []
        return [{"name": "Tab_3", "content": "https://tensorboard.dev/experiment/8m1aX0gcQ7aEmH0J7kbBtg/#scalars"}]


app = L.LightningApp(Flow(), flow_cloud_compute=L.CloudCompute("gpu"))