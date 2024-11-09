# from modeling_mask2former import Mask2FormerForUniversalSegmentation
from transformers import AutoConfig
import torch
import numpy as np
from transformers import AutoConfig,Mask2FormerConfig,Mask2FormerModel,Mask2FormerImageProcessor, Trainer, TrainingArguments
# from modeling_mask2former import Mask2FormerForUniversalSegmentation
from torch.utils import data
from torch import nn
import os

from datasets import load_dataset
import torchvision.transforms as tfs

from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate import DistributedDataParallelKwargs
from models.upernet import UperNetForSemanticSegmentation

import argparse

ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

class ChangeDetectionDataset(data.Dataset):
    def __init__(self,dataset,transform=None) -> None:
        super().__init__()
        self.dataset=dataset
        self.transform=transform
    def __len__(self):
        return(len(self.dataset))
    def __getitem__(self, index):
        imageA=self.transform(self.dataset[index]["imageA"])
        imageB=self.transform(self.dataset[index]["imageB"])
        label=tfs.ToTensor()(self.dataset[index]["label"])
        label=torch.cat([label],dim=0)
        return imageA,imageB,label

def main(args):
    batch_size=args.batch_size
    backbone_type=args.backbone_type

    # backbone_type="ResNet_Diff_50"
    # backbone_type="Swin_Diff_T"
    # backbone_type="VSSM_T_ST_Diff"

    logger = get_logger(__name__)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device=accelerator.device
    batch_size=16
    dataset=load_dataset("HZDR-FWGEL/MineNetCD256")
    logger.info(dataset,main_process_only=True)
    train_ds=dataset["train"]
    test_ds=dataset["test"]
    val_ds=dataset["val"]
    transform=tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean=ADE_MEAN,std=ADE_STD),
    ])

    train_dataset=ChangeDetectionDataset(train_ds,transform=transform)
    val_dataset=ChangeDetectionDataset(val_ds,transform=transform)
    test_dataset=ChangeDetectionDataset(test_ds,transform=transform)

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    channel_mixing=args.channel_mixing
    # VMamba here
    if "VSSM" in backbone_type:
        pretrained_model_name ="openmmlab/upernet-swin-tiny"
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.update({"num_labels":2,"Backbone_type":backbone_type, "channel_mixing":channel_mixing})
        model=UperNetForSemanticSegmentation._from_config(config)

    # Swin Transformer here
    elif "Swin" in backbone_type:
        pretrained_model_name ="openmmlab/upernet-swin-base"
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.update({"num_labels":2,"Backbone_type":backbone_type, "channel_mixing":channel_mixing})
        model=UperNetForSemanticSegmentation.from_pretrained(pretrained_model_name, config=config,ignore_mismatched_sizes=True)
    elif "ResNet" in backbone_type:
        pretrained_model_name ="openmmlab/upernet-swin-base"
        config = AutoConfig.from_pretrained(pretrained_model_name)
        config.update({"num_labels":2,"Backbone_type":backbone_type, "channel_mixing":channel_mixing})
        model=UperNetForSemanticSegmentation._from_config(config)
    else:
        print("We support Swin, ResNet or Vmamba")


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=1e-7, last_epoch=-1, verbose=False)
    running_loss = 0.0
    num_samples = 0

    model.to(device)

    model, optimizer, train_dataloader, scheduler=accelerator.prepare(model, optimizer, train_dataloader, scheduler)
    
    for epoch in range(args.epochs):
        logger.info(f'Epoch:{epoch}',main_process_only=True)
        model.train()
        for idx, batch in enumerate(tqdm(train_dataloader,disable=not accelerator.is_local_main_process, miniters=50)):
            # Reset the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            
            imageA, imageB, labels=batch
            pixel_values=torch.cat([imageA,imageB],dim=0)

            outputs = model(
                    pixel_values=pixel_values,
                    labels=labels.squeeze().long(),
                )
            # Backward propagation
            loss = outputs.loss
            accelerator.backward(loss)

            batch_size = labels.size(0)
            running_loss += loss.item()
            num_samples += batch_size

            if idx % 100 == 0:
                print("Loss:", running_loss/num_samples)

            # Optimization
            optimizer.step()
            scheduler.step()
        
        if (epoch+1) // 5 ==0:
            if channel_mixing:
                save_pretrained_path=f"./exp/minenetcd_upernet_{backbone_type}_Pretrained_ChannelMixing_Dropout/{epoch}"
            else:
                save_pretrained_path=f"./exp/minenetcd_upernet_{backbone_type}_Pretrained/{epoch}"
            os.makedirs(save_pretrained_path,exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(save_pretrained_path)
    if args.push_to_hub:
        if channel_mixing:
            push_to_hub_path=f"minenetcd_upernet_{backbone_type}_Pretrained_ChannelMixing_Dropout"
        else:
            push_to_hub_path=f"minenetcd_upernet_{backbone_type}_Pretrained"
        accelerator.unwrap_model(model).push_to_hub(save_pretrained_path)

def args():
    parser = argparse.ArgumentParser(description='MineNetCD Training Arguments')

    parser.add_argument('--batch-size', type=int, default=8, help='batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--backbone-type', type=str, default='ResNet_Diff_50', choices=['ResNet_Diff_18','ResNet_Diff_50','ResNet_Diff_101','Swin_Diff_T', 'Swin_Diff_S', 'Swin_Diff_B', 'VSSM_T_ST_Diff', 'VSSM_S_ST_Diff', 'VSSM_B_ST_Diff'], help='Backbone Type (Modularized Encoder)')
    parser.add_argument('--channel-mixing', type=bool, default=False, help='whether using ChangeFFT.')
    parser.add_argument('--push-to-hub', type=bool, default=False, help='whether pushing trained models to your huggingface repo, you need to login before using this feature.')
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args=args()
    main(args)

