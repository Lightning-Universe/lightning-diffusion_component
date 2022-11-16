# https://github.com/Lightning-AI/stable-diffusion-deploy/blob/a560875612548f531972f71921c5a09a1fe1fb74/muse/components/
# https://github.com/Lightning-AI/stable-diffusion-deploy/blob/a560875612548f531972f71921c5a09a1fe1fb74/muse/components/stable_diffusion_serve.py
import os
from typing import List, Optional

import lightning as L
import torch
from lightning import BuildConfig, LightningWork
from lightning.app.storage import Drive
from torch.utils.data import DataLoader, Dataset


NSFW_PROMPTS = [
    "nudity",
    "porn",
    "sex",
    "hentai",
    "fetish pornography",
    "violence",
    "gore",
    "murder",
    "blood",
    "guns",
    "profanity",
    "racist",
    "homophobic",
    "sexist",
    "swastika",  # Nazi Symbol
    "penis",
    "anus",
    "buttocks",
    "breasts",
    "dildo",
    "genitals",
    "nipples",
    "vagina",
    "rectum",
]


class TextPromptDataset(Dataset):
    def __init__(self, prompts):
        super().__init__()
        self.prompts = prompts

    def __getitem__(self, ix):
        return self.prompts[ix]

    def __len__(self):
        return len(self.prompts)


class SafetyCheckerBuildConfig(BuildConfig):
    def build_commands(self) -> List[str]:
        return ["python -m pip install git+https://github.com/openai/CLIP.git"]


class SafetyCheckerEmbedding(LightningWork):
    def __init__(self, nsfw_list: Optional[List] = None, drive: Optional[Drive] = None):
        super().__init__(
            parallel=False, cloud_compute=L.CloudCompute("cpu-medium"), cloud_build_config=SafetyCheckerBuildConfig()
        )

        self.drive = drive
        self.safety_embeddings_filename = "safety_embedding.pt"

    def run(self):
        import clip as openai_clip

        model, _ = openai_clip.load("ViT-B/32", device="cpu")
        ds = TextPromptDataset(NSFW_PROMPTS)
        dl = DataLoader(ds, shuffle=False, batch_size=4, num_workers=os.cpu_count())

        encoded_text = []
        for batch in dl:
            text_features = model.encode_text(openai_clip.tokenize(batch))
            encoded_text.append(text_features)

        encoded_text = torch.vstack(encoded_text)
        encoded_text = torch.nn.functional.normalize(encoded_text, p=2, dim=1)
        torch.save(encoded_text, self.safety_embeddings_filename)
        self.drive.put(self.safety_embeddings_filename)


class SafetyChecker:
    """The default safety checker.

    >>> safety_checker = SafetyChecker(self.safety_embeddings_filename)
    >>> pil_results = lit_model(some_inputs)
    >>> nsfw_content = safety_checker(pil_results)
    >>> for i, nsfw in enumerate(nsfw_content):
            if nsfw:
                pil_results[i] = Image.open("assets/nsfw-warning.png")
    """
    def __init__(self, embeddings_path):
        import clip as openai_clip

        self.model, self.preprocess = openai_clip.load("ViT-B/32", device="cpu")
        self.text_embeddings = torch.load(embeddings_path)

    def __call__(self, images):
        images = torch.stack([self.preprocess(img) for img in images])
        encoded_images = self.model.encode_image(images)

        encoded_images = torch.nn.functional.normalize(encoded_images, p=2, dim=1)
        similarity = torch.mm(encoded_images, self.text_embeddings.transpose(0, 1))
        return torch.any(similarity > 0.24, dim=1).tolist()
