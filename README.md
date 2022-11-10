This app dowload the data from a google drive foilder, then creates the priors (for testing this priors are the result of 4 denoising steps), trains the model (for testing I'm just runing one epoch) and launches a gradio server with the retrained model.

__________
# Usage:
```
# Using cpu-medium
lightning run app app.py --cloud

# Using GPU
lightning run app demo_app.py --cloud --env LIGHTNING_JUPYTER_LAB_COMPUTE=gpu
```
