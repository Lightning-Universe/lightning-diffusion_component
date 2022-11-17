import lightning as L
from lightning.lite import LightningLite
from typing import List, Optional
from datasets import DreamBoothDataset, PromptDataset
import os
import torch
from diffusers import StableDiffusionPipeline
import requests
import gc
import torch.nn.functional as F
from functools import partial
from lightning.app.storage import Drive
from dataclasses import dataclass


def collate_fn(examples, tokenizer, preservation_prompt):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]

    # Concat class and instance examples for prior preservation.
    # We do this to avoid doing two forward passes.
    if preservation_prompt:
        input_ids += [example["class_prompt_ids"] for example in examples]
        pixel_values += [example["class_images"] for example in examples]

    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
        {"input_ids": input_ids},
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch


@dataclass
class DreamBoothTuner:

    image_urls: List[str]
    prompt: str
    preservation_prompt: str
    validation_prompt: str
    num_preservation_images: int = 100
    pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4"
    revision: Optional[str] = "fp16"
    tokenizer_name: Optional[str] = None
    max_steps: int = 5
    prior_loss_weight: float = 1
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-6
    lr_scheduler = "constant"
    lr_warmup_steps: int = 0
    precision: int = 16
    use_auth_token: str = "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"
    seed: int = 42
    gradient_checkpointing: bool = True
    resolution: int = 512
    scale_lr: bool = True
    center_crop: bool = True

    """
    The `DreamBoothFineTuner` fine-tunes stable diffusion models using the methodology introduced in

    DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation
    https://arxiv.org/abs/2208.12242

    The code is highly inspired from the diffusers `train_dreambooth.py` script:
    https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/train_dreambooth.py.

    Arguments:
        image_urls: List of image urls to fine-tune the models.
        prompt: The prompt to run the fine-tuning process.
            The prompt needs to be in the format of the reference with the `[...]`.
            Reference Format: `A photo of [NOUN] [DESCRIPTIVE CLASS] [DESCRIPTION FOR THE NEW GENERATED IMAGES]`.
            Example: `A photo of [peter] [cat clay toy] [riding a bicycle]`.
        num_preservation_images: The number of images to preserve the model abilities.
        pretrained_model_name_or_path: The name of the model.
        revision: The revision commit for the model weights.
        tokenizer_name: The name of the tokenizer.
        prior_loss_weight: The weight of prior preservation loss to preserve knowledge.
        train_batch_size: The batch size used during training.
        gradient_accumulation_steps: Number of training batch before updating the weights.
        learning_rate: The learning rate to optimize the model.
        lr_scheduler: The LR scheduler to be used.
        lr_warmup_steps: The number of warmup steps.
        precision: The precision to be used for fine-tuning the model.
        seed: The seed to initialize the random initializers.
        resolution: The resolution of the image to train upon.
        center_crop: Whether to crop the images in the center
    """
    @property
    def user_images_data_dir(self) -> str:
        return os.path.join(os.getcwd(), "data", 'user_images')

    @property
    def preservation_images_data_dir(self) -> str:
        return os.path.join(os.getcwd(), "data", 'preservation_images')

    @property
    def validation_images_data_dir(self) -> str:
        return os.path.join(os.getcwd(), "data", 'validation_images')

    def run(self, model: StableDiffusionPipeline):
        lite = LightningLite(precision=self.precision, strategy="deepspeed_stage_2_offload")

        self.setup(lite, model)

        optimizer, dtype = self.prepare_model(lite, model)

        train_dataloader = self.prepare_data(lite, model)

        for step, batch in enumerate(train_dataloader):

            if self.gradient_accumulation_steps > 1:
                is_accumulating = step % self.gradient_accumulation_steps != 0
            else:
                is_accumulating = False

            with lite.no_backward_sync(model.unet, enabled=is_accumulating):

                with torch.no_grad():
                    # Convert images to latent space
                    latents = model.vae.encode(batch["pixel_values"].to(lite.device, dtype=dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents, device=lite.device, dtype=dtype)
                bsz = latents.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(0, model.scheduler.config.num_train_timesteps, (bsz,), device=lite.device).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = model.scheduler.add_noise(latents, noise, timesteps)

                with torch.no_grad():
                    # Get the text embedding for conditioning
                    encoder_hidden_states = model.text_encoder(batch["input_ids"].to(lite.device))[0]

                # Predict the noise residual
                noise_pred = model.unet(noisy_latents, timesteps, encoder_hidden_states).sample

                if self.preservation_prompt:
                    # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                    noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                    noise, noise_prior = torch.chunk(noise, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                    # Compute prior loss
                    prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + self.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                lite.backward(loss)

            # Step the optimizer every 8 batches
            if not is_accumulating:
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            if lite.is_global_zero:
                print(f"Step {step}/{self.max_steps}: {loss}")

        torch.distributed.destroy_process_group()

        if lite.is_global_zero:

            self.evaluate_model(model)

            # TODO: Implement DeepSpeed saving in Lite.
            model.unet = model.unet.module

            model.save_pretrained("model.pt")

            # drive = Drive("lit://models", component_name="unet")
            # drive.put("model.pt")

        print("Dreambooth finetuning is done!")

    def setup(self, lite, model):
        if self.preservation_prompt is None:
            return

        if lite.local_rank != 0:
            return

        self._download_images()

        self._generate_preservation_images(lite, model)

    def prepare_model(self, lite, model):
        if self.seed is not None:
            L.seed_everything(self.seed)

        # Freeze the var and text encoder
        model.vae.requires_grad_(False)
        model.text_encoder.requires_grad_(False)

        if self.gradient_checkpointing:
            model.unet.enable_gradient_checkpointing()
            model.text_encoder.gradient_checkpointing_enable()

        world_size = int(os.getenv("WORLD_SIZE", 1))

        if self.scale_lr:
            self.learning_rate = (
                self.learning_rate * self.gradient_accumulation_steps * self.train_batch_size * world_size
            )

        # # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        optimizer_class = torch.optim.AdamW

        params_to_optimize = model.unet.parameters()
        optimizer = optimizer_class(
            params_to_optimize,
            lr=self.learning_rate,
        )

        model.unet, optimizer = lite.setup(model.unet, optimizer)  # Scale your model / optimizers

        dtype = torch.float32
        if self.precision == 16:
            dtype = torch.float16
        elif self.precision == "bf16":
            dtype = torch.bfloat16

        model.vae = model.vae.to(device=lite.device, dtype=dtype)
        model.text_encoder = model.text_encoder.to(device=lite.device, dtype=dtype)

        model.unet.train()

        return optimizer, dtype

    def prepare_data(self, lite: LightningLite, model):
        train_dataset = DreamBoothDataset(
            instance_data_root=self.user_images_data_dir,
            instance_prompt=self.prompt,
            class_data_root=self.preservation_images_data_dir if self.preservation_prompt else None,
            class_prompt=self.preservation_prompt,
            tokenizer=model.tokenizer,
            size=self.resolution,
            center_crop=self.center_crop,
            length=self.max_steps,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            collate_fn=partial(collate_fn, tokenizer=model.tokenizer, preservation_prompt=self.preservation_prompt),
            num_workers=1,
        )

        return lite.setup_dataloaders(train_dataloader)

    def _download_images(self):
        """Download the images provided by the user"""

        os.makedirs(self.user_images_data_dir, exist_ok=True)

        L = len(os.listdir(self.user_images_data_dir))

        for idx, image_url in enumerate(self.image_urls):

            r = requests.get(image_url, stream=True)

            if r.status_code == 200:

                path = os.path.join(self.user_images_data_dir, f"{idx + L}.jpg")

                with open(path, 'wb') as f:
                    for chunk in r.iter_content(1024):
                        f.write(chunk)
            else:
                print(f"The image from {image_url} doesn't exist.")

    def _generate_preservation_images(self, lite: LightningLite, model):

        os.makedirs(self.preservation_images_data_dir, exist_ok=True)

        sample_dataset = PromptDataset(self.preservation_prompt, self.num_preservation_images)
        sample_dataloader = torch.utils.data.DataLoader(
            sample_dataset,
            batch_size=2,
        )

        sample_dataloader = lite.setup_dataloaders(sample_dataloader)
        model.to(lite.device)

        L = len(os.listdir(self.preservation_images_data_dir))

        counter = 0

        for example in sample_dataloader:

            if (L + counter) >= self.num_preservation_images:
                break

            with torch.inference_mode():
                images = model(example["prompt"]).images

            for image in images:
                path = os.path.join(self.preservation_images_data_dir, f"{counter + L}.jpg")
                image.save(path)
                counter += 1

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()

    def evaluate_model(self, model):
        os.makedirs(self.validation_images_data_dir, exist_ok=True)

        counter = 0 

        model.vae = model.vae.to(torch.float32)
        model.text_encoder = model.text_encoder.to(torch.float32)
        model.safety_checker = None

        with torch.inference_mode():
            images = model(self.validation_prompt, num_images_per_prompt=10).images

            for image in images:
                path = os.path.join(self.validation_images_data_dir, f"{counter}.jpg")
                image.save(path)
                counter += 1
