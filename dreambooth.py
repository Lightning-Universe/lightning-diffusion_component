import lightning as L
from lightning.lite import LightningLite
from typing import List, Optional
from datasets import DreamBoothDataset, PromptDataset
import os
from pathlib import Path
import torch
import hashlib
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel



class DreamBoothFineTuner(L.LightningWork):

    def __init__(
        self,
        image_urls: List[str],
        prompt: str,
        preservation_prompt: Optional[str] = None,
        pretrained_model_name_or_path: str = "CompVis/stable-diffusion-v1-4",
        revision: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        prior_loss_weight: float = 1.0,
        train_batch_size: int = 1,
        gradient_accumulation_steps: int = 1,
        learning_rate: float = 5e-6,
        lr_scheduler = "constant",
        lr_warmup_steps: int = 0,
        max_train_steps: int = 400,
        precision: int = 16,
        cloud_compute: L.CloudCompute = L.Compute("gpu"),
        hf_token: str = "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",
        seed: int = 42,
        **kwargs,
    ):
        """
        The `DreamBoothFineTuner` fine-tunes stable diffusion models using the methodology introduced in

        DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation
        https://arxiv.org/abs/2208.12242

        Arguments:
            image_urls: List of image urls to fine-tune the models.
            prompt: The prompt to describe the images.
                Example: `A [V] dog` where `[V]` is a special name given for
                the diffusion model to learn about this new concept.
            preservation_prompt: The prompt used for the diffusion model to preserve knowledge.
                Example: `A dog`
            pretrained_model_name_or_path: The name of the model.
            revision: The revision commit for the model weights.
            tokenizer_name: The name of the tokenizer.
            prior_loss_weight: The weight of prior preservation loss to preserve knowledge.
            train_batch_size: The batch size used during training.
            gradient_accumulation_steps: Number of training batch before updating the weights.
            learning_rate: The learning rate to optimize the model.
            lr_scheduler: The LR scheduler to be used.
            lr_warmup_steps: The number of warmup steps.
            max_train_steps: The number of training steps to fine-tune the model.
            precision: The precision to be used for fine-tuning the model.
            seed: The seed to initialize the random initializers.
            kwargs: The keywords arguments passed down to the work.
        """
        super().__init__(cloud_compute=cloud_compute, parallel=True, **kwargs)

        self.image_urls = image_urls
        self.prompt = prompt
        self.preservation_prompt = preservation_prompt
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.revision = revision
        self.tokenizer_name = tokenizer_name
        self.prior_loss_weight = prior_loss_weight
        self.train_batch_size = train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.lr_warmup_steps = lr_warmup_steps
        self.max_train_steps = max_train_steps
        self.precision = precision
        self.hf_token = hf_token
        self.seed = seed

    def run(self):

        # Used to force Lite to setup the distributed environnement.
        # TODO: Remove this.
        os.environ["LT_CLI_USED"] = "1"

        lite = LightningLite(precision=self.precision)

        if self.seed is not None:
            L.seed_everything(self.seed)

        self.prepare_data(lite)

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.push_to_hub:
                if args.hub_model_id is None:
                    repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
                else:
                    repo_name = args.hub_model_id
                repo = Repository(args.output_dir, clone_from=repo_name)

                with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
            elif args.output_dir is not None:
                os.makedirs(args.output_dir, exist_ok=True)

        # Load the tokenizer
        if args.tokenizer_name:
            tokenizer = CLIPTokenizer.from_pretrained(
                args.tokenizer_name,
                revision=args.revision,
            )
        elif args.pretrained_model_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(
                args.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=args.revision,
            )

        # Load models and create wrapper for stable diffusion
        text_encoder = CLIPTextModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="vae",
            revision=args.revision,
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.revision,
        )

        vae.requires_grad_(False)
        if not args.train_text_encoder:
            text_encoder.requires_grad_(False)

        if args.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
            if args.train_text_encoder:
                text_encoder.gradient_checkpointing_enable()

        if args.scale_lr:
            args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
            )

        # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        params_to_optimize = (
            itertools.chain(unet.parameters(), text_encoder.parameters()) if args.train_text_encoder else unet.parameters()
        )
        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        noise_scheduler = DDPMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

        train_dataset = DreamBoothDataset(
            instance_data_root=args.instance_data_dir,
            instance_prompt=args.instance_prompt,
            class_data_root=args.class_data_dir if args.with_prior_preservation else None,
            class_prompt=args.class_prompt,
            tokenizer=tokenizer,
            size=args.resolution,
            center_crop=args.center_crop,
        )

        def collate_fn(examples):
            input_ids = [example["instance_prompt_ids"] for example in examples]
            pixel_values = [example["instance_images"] for example in examples]

            # Concat class and instance examples for prior preservation.
            # We do this to avoid doing two forward passes.
            if args.with_prior_preservation:
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

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        if args.train_text_encoder:
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, text_encoder, optimizer, train_dataloader, lr_scheduler
            )
        else:
            unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                unet, optimizer, train_dataloader, lr_scheduler
            )

        weight_dtype = torch.float32
        if args.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif args.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move text_encode and vae to gpu.
        # For mixed precision training we cast the text_encoder and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        vae.to(accelerator.device, dtype=weight_dtype)
        if not args.train_text_encoder:
            text_encoder.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if overrode_max_train_steps:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("dreambooth", config=vars(args))

        # Train!
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
        progress_bar.set_description("Steps")
        global_step = 0

        for epoch in range(args.num_train_epochs):
            unet.train()
            if args.train_text_encoder:
                text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(unet):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    if args.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior.float(), noise_prior.float(), reduction="mean")

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                    else:
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = (
                            itertools.chain(unet.parameters(), text_encoder.parameters())
                            if args.train_text_encoder
                            else unet.parameters()
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            accelerator.wait_for_everyone()

        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet),
                text_encoder=accelerator.unwrap_model(text_encoder),
                revision=args.revision,
            )
            pipeline.save_pretrained(args.output_dir)

            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)

        accelerator.end_training()


    def prepare_data(self, lite):
        if self.preservation_prompt is None:
            return

        if lite.local_rank != 0:
            return

        self._download_images()

        self._generate_preservation_images(lite)

    @staticmethod
    def _download_images():
        """TODO"""

    def _generate_preservation_images(self, lite: LightningLite):
        preservation_images = os.path.join(os.getcwd(), "data", 'preservation_images')

        os.makedirs(preservation_images, exist_ok=True)

        pipeline = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
            revision=self.revision,
        )

        num_new_images = args.num_class_images - cur_class_images

        sample_dataset = PromptDataset(args.class_prompt, num_new_images)
        sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=args.sample_batch_size)

        sample_dataloader = lite.prepare(sample_dataloader)
        pipeline.to(lite.device)

        for example in sample_dataloader:

            images = pipeline(example["prompt"]).images

            for i, image in enumerate(images):
                hash_image = hashlib.sha1(image.tobytes()).hexdigest()
                image_filename = class_images_dir / f"{example['index'][i] + cur_class_images}-{hash_image}.jpg"
                image.save(image_filename)

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()