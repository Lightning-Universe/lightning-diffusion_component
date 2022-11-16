
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

import torch
from pytorch_lightning.lite import LightningLite
from diffusers import AutoencoderKL, DDPMScheduler, PNDMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import torch.nn.functional as F
from datasets import DreamBoothDataset

from tqdm.auto import tqdm
import os
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import lightning as L




class Lite(LightningLite):
    

    def run(self, unet,tokenizer,vae,text_encoder,args,dev):

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

        # Enable Checkpointing on inet to reduce memory usage
        unet.enable_gradient_checkpointing() 


        # define optimezer
        optimizer = torch.optim.AdamW(
                    unet.parameters(),  # only optimize unet
                    lr=args.learning_rate)

        unet, optimizer = self.setup(unet, optimizer)  # Scale your model / optimizers


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
        unet.train()


        # time_vae = 0
        # time_noise_s = 0
        # time_encoder  = 0
        # time_unet  = 0
        # time_bckp = 0
        # time_step = 0

        # move to GPU 
        vae = vae.to(dev)
        text_encoder = text_encoder.to(dev)

        for epoch in range(args.num_epochs):
            with tqdm(dataloader, unit="batch",desc="Training") as tepoch:
                for i,batch in enumerate(tepoch):
                    optimizer.zero_grad()
                    
                    #Initial_memory = torch.cuda.memory_allocated(dev)
                    # Convert images to latent space
                    with torch.no_grad():
                        # start_time = time.time()
                        latents = vae.encode(batch["pixel_values"])[0]
                        latents=latents.sample()
                        # time_vae += (time.time() - start_time)
                        latents = latents * 0.18215
                    #memory_consumed_vae = Initial_memory-torch.cuda.memory_allocated(dev)
                   

                    bsz = latents.shape[0]

                    # Sample a random timestep for each image
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                    ).long()

                    # Get the text embedding for conditioning
                    #Initial_memory = torch.cuda.memory_allocated(dev)
                    with torch.no_grad():
                        # start_time = time.time()
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                        # time_encoder += (time.time() - start_time)
                    
                    #memory_consumed_enc = Initial_memory-torch.cuda.memory_allocated(dev)
                    

                    # Sample noise that we'll add to the latents

                    #start_time = time.time()

                    # generate random noise
                    noise = torch.randn(latents.shape).to(latents.device)

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    #time_noise_s += (time.time() - start_time)

                    
                    # Predict the noise residual

                    #start_time = time.time()
                    #Initial_memory = torch.cuda.memory_allocated(dev)

                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states)["sample"]

                    # memory_consumed_unet = torch.cuda.memory_allocated(dev)-Initial_memory
                    # time_unet += (time.time() - start_time)
        
                    if args.with_prior_preservation:
                        # Chunk the noise and noise_pred into two parts and compute the loss on each part separately.
                        noise_pred, noise_pred_prior = torch.chunk(noise_pred, 2, dim=0)
                        noise, noise_prior = torch.chunk(noise.to(dev), 2, dim=0)

                        # Compute instance loss
                        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                        # Compute prior loss
                        prior_loss = F.mse_loss(noise_pred_prior, noise_prior, reduction="none").mean([1, 2, 3]).mean()
                        
                        noise_pred = None
                        noise_pred_prior = None

                        # Add the prior loss to the instance loss.
                        loss = loss + args.prior_loss_weight * prior_loss
                        print(loss)
                    else:
                        loss = F.mse_loss(noise_pred, noise, reduction="none").mean([1, 2, 3]).mean()

                    # backporpagation
                    #start_time = time.time()
                    #Initial_memory = torch.cuda.memory_allocated(dev)

                    self.backward(loss)  # instead of loss.backward()

                    # memory_consumed_bckp = Initial_memory-torch.cuda.memory_allocated(dev)
                    # time_bckp += (time.time() - start_time)
                    
                    # Initial_memory = torch.cuda.memory_allocated(dev)
                    # start_time = time.time()

                    optimizer.step()
                    
                    # time_step += (time.time() - start_time)
                    # memory_consumed_step = Initial_memory-torch.cuda.memory_allocated(dev)

                    tepoch.set_postfix(loss=loss.item())

                    # update tenorboard
                    # writer.add_scalar("Train/Loss", loss, i)
                    # writer.close()


                # num = len(tepoch)*(epoch+1)
                # file = open('times.json', 'w+')
                # data = {"time_unet":time_unet/num,
                #         "time_vae":time_vae/num,
                #         "time_encoder":time_encoder/num,
                #         "time_bckp":time_bckp/num,
                #         "time_step":time_step/num,
                #         "time_noise_s":time_noise_s/num,
                #         "number_steps": num}
                # json.dump(data, file)


                # file = open('memory.json', 'w+')
                # data = {"memory_unet":memory_consumed_unet,
                #         "memory_vae":memory_consumed_vae,
                #         "memory_encoder":memory_consumed_enc,
                #         "memory_bckp":memory_consumed_bckp,
                #         "memory_step":memory_consumed_step}
                # json.dump(data, file)

                return



class Training(L.LightningWork):  
    def __init__(self,cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.data_dir = None
        self.prior_dir = None
        self.checkpoint_dir = None

    def run(self,data_dir,prior_dir,args): 

        self.data_dir = data_dir
        self.prior_dir = prior_dir
        self.checkpoint_dir = "lit://outputs/dreambooth-concept"
        os.makedirs( self.checkpoint_dir, exist_ok=True)

        # type of devie
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
         # parameter to download from HF (Need to change this)
        pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5" #"CompVis/stable-diffusion-v1-4"
        YOUR_TOKEN="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"
    
        
        # get the models
        vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae",
                                    use_auth_token=YOUR_TOKEN)

        # Load the tokenizer and text encoder to tokenize and encode the text. 
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        # The UNet model for generating the latents.
        unet = UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet",
                                                    use_auth_token=YOUR_TOKEN)


        # TENSOR BOAR LAUNCHER
        # log_dir = "runs"
        # self._process = Popen(
        #             f"tensorboard --logdir='{log_dir}' --host {self.host} --port {self.port}",
        #             shell=True)


        # update folders in the  arguments
        args.class_data_dir = self.data_dir
        args.instance_data_dir= self.prior_dir
        
        
        strategy_training = "deepspeed_stage_2_offload" if torch.cuda.is_available() else "ddp"
        
        # trianing
        Lite( devices="auto", accelerator="auto", 
            strategy=strategy_training).run(unet,tokenizer,vae,text_encoder,args,device)
        
        # update pipeline and save it currently in HF
        pipeline = StableDiffusionPipeline(
                        text_encoder=text_encoder,
                        vae=vae,
                        unet=unet,
                        tokenizer=tokenizer,
                        scheduler=PNDMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True),
                        safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"),
                        feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32"),)
        
        pipeline.save_pretrained(self.checkpoint_dir) 
        print("DONE training ")






      
    




   




    