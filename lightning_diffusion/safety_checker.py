# https://github.com/Lightning-AI/stable-diffusion-deploy/blob/a560875612548f531972f71921c5a09a1fe1fb74/muse/components/
# https://github.com/Lightning-AI/stable-diffusion-deploy/blob/a560875612548f531972f71921c5a09a1fe1fb74/muse/components/stable_diffusion_serve.py
import torch
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


class DefaultSafetyFilter:
    """The default safety filter.

    >>> safety_filter = SafetyFilter()
    >>> safety_filter.setup()
    >>> nsfw_content = safety_checker(pil_results)
    >>> for i, nsfw in enumerate(nsfw_content):
            if nsfw:
                ...
    """
    def __init__(self) -> None:
        import clip as openai_clip

        self.model, self.preprocess = openai_clip.load("ViT-B/32", device="cpu")
        self.prepare_nsfw_embeddings()

    def __call__(self, images):
        images = torch.stack([self.preprocess(img) for img in images])
        encoded_images = self.model.encode_image(images)

        encoded_images = torch.nn.functional.normalize(encoded_images, p=2, dim=1)
        similarity = torch.mm(encoded_images, self.nsfw_embeddings.transpose(0, 1))
        return torch.any(similarity > 0.24, dim=1).tolist()

    def prepare_nsfw_embeddings(self) -> None:
        import clip as openai_clip

        ds = TextPromptDataset(NSFW_PROMPTS)
        dl = DataLoader(ds, shuffle=False, batch_size=4)

        encoded_text = []
        for batch in dl:
            text_features = self.model.encode_text(openai_clip.tokenize(batch))
            encoded_text.append(text_features)

        encoded_text = torch.vstack(encoded_text)
        self.nsfw_embeddings = torch.nn.functional.normalize(encoded_text, p=2, dim=1)
