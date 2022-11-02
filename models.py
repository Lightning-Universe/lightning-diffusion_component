
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import torch
from pytorch_lightning.lite import LightningLite
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import torch.nn.functional as F
from datasets import DreamBoothDataset

from tqdm.auto import tqdm
from argparse import Namespace
import os
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import lightning as L
from lightning.app.storage import Drive
from lightning.app.storage.path import Path
from subprocess import Popen
from PIL import Image


class Lite(LightningLite):
    

    def run(self, unet,tokenizer,vae,text_encoder,args):
        # concatenating fuction
        def collate_fn(examples):
            # get the description of the samples
            input_ids = [example["instance_prompt_ids"] for example in examples]

            # get images
            pixel_values = [example["instance_images"] for example in examples]

            # concat class and instance examples for prior preservation
            if args.with_prior_preservation:
                input_ids += [example["class_prompt_ids"] for example in examples]
                pixel_values += [example["class_images"] for example in examples]

            # concatenated pictures
            pixel_values = torch.stack(pixel_values)
            pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

            # concatenated prompts
            input_ids = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt").input_ids

            # batch
            batch = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
            }
            return batch
        
        # start tensorboard
        # writer = SummaryWriter('runs')

        # define model
        model = unet

        # define optimezer
        optimizer = torch.optim.AdamW(
        model.parameters(),  # only optimize unet
        lr=args.learning_rate)
        model, optimizer = self.setup(model, optimizer)  # Scale your model / optimizers


        # noise scheduler
        noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
        
        # Dataset
        train_dataset = DreamBoothDataset(
                        instance_data_root=args.instance_data_dir,
                        instance_prompt=args.instance_prompt,
                        class_data_root=args.class_data_dir if args.with_prior_preservation else None,
                        class_prompt=args.class_prompt,
                        tokenizer=tokenizer,
                        size=args.resolution,
                        center_crop=args.center_crop,)

        # Dataloader
        dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)
        dataloader = self.setup_dataloaders(dataloader)  # Scale your dataloaders

        # start training
        model.train()

        for epoch in range(args.num_epochs):
            with tqdm(dataloader, unit="batch",desc="Training") as tepoch:
                for i,batch in enumerate(tepoch):
                    optimizer.zero_grad()

                    # Convert images to latent space
                    with torch.no_grad():
                        latents = vae.encode(batch["pixel_values"]).sample()
                        latents = latents * 0.18215

                    # Sample noise that we'll add to the latents
                    noise = torch.randn(latents.shape).to(latents.device)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()

                    # Get the text embedding for conditioning
                    with torch.no_grad():
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                    
                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Predict the noise residual
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]
                    print("NOISE",noise_pred.shape)
                    if args.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise, 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2, 3]).mean()

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                        print(loss)
                    else:
                        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                    # backporpagation
                    self.backward(loss)  # instead of loss.backward()
                    optimizer.step()
                    tepoch.set_postfix(loss=loss.item())

                    # update tenorboard
                    # writer.add_scalar("Train/Loss", loss, i)
                    # writer.close()
                return



class Training(L.LightningWork):  
    def __init__(self,cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.drive_3 = Drive("lit://drive_3")

    def run(self,args): 

        # type of devie
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
         # parameter to download from HF (Need to change this)
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4" #@param {type:"string"}
        YOUR_TOKEN="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"
    
        
        # get the models
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder",use_auth_token=YOUR_TOKEN)
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae",use_auth_token=YOUR_TOKEN)
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet",use_auth_token=YOUR_TOKEN)  
        tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path,subfolder="tokenizer",use_auth_token=YOUR_TOKEN)

        # TENSOR BOAR LAUNCHER
        # log_dir = "runs"
        # self._process = Popen(
        #             f"tensorboard --logdir='{log_dir}' --host {self.host} --port {self.port}",
        #             shell=True)


        # make local folders for training 
        if not os.path.exists("class_data_dir"):
            os.makedirs("class_data_dir")

        if not os.path.exists("my_concept"):
            os.makedirs("my_concept")
        
        # get the raining examples from the drive
        files = args.class_data_dir.list(".")
        for i,f in enumerate(files):
            args.class_data_dir.get(f)
            img = Image.open(f)
            img.save(f"class_data_dir/{i}.jpg")

        files = args.instance_data_dir.list(".")
        for i,f in enumerate(files):
            args.instance_data_dir.get(f)
            img = Image.open(f)
            img.save(f"my_concept/{i}.jpeg")


        # update folders in the  arguments
        args.class_data_dir = "class_data_dir"
        args.instance_data_dir = "my_concept"

        # trianing
        Lite(strategy="ddp", devices=1, accelerator=device, precision="bf16").run(unet,tokenizer,vae,text_encoder,args)
        
        # update pipeline and save it currently in HF
        pipeline = StableDiffusionPipeline(
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet,
                        tokenizer=tokenizer,
                        scheduler=PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True),
                        safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),)
        pipeline.save_pretrained(args.output_dir) 

         # save files to the cloud to use in the deployment
        files = list(Path("dreambooth-concept").iterdir())
        for i in files:
            if os.path.isfile(i):
                self.drive_3.put(str(i))
            else:
                for j in list(i.iterdir()):
                    self.drive_3.put(str(j))






      
    




   




    