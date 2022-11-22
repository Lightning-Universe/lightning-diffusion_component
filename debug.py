from lightning.app.runners import MultiProcessRuntime

from serve_diffusion_component import app

if __name__ == "__main__":
    MultiProcessRuntime(app, start_server=True, host="0.0.0.0", port=7502).dispatch()
