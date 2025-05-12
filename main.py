import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import numpy as np
import pytorch_lightning as pl
from model import PromptIR
import torch.nn.functional as F
from torchvision import transforms
import random




def main(
    dataset_root="hw4_realse_dataset",
    batch_size=2,
    lr_max=2e-4,
    lr_min=5e-6,
    epoches=20,
    val_split=0.1,
    gradient_clip_value=0.8,
    num_workers=4 if os.name != "nt" else 0,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
):
    train_loader, val_loader = get_dataloaders(
        dataset_root, batch_size, val_split, num_workers
    )
    model = PromptIRLightning(lr_max=lr_max, lr_min=lr_min, T_max=epoches)
    trainer = pl.Trainer(
        max_epochs=epoches,
        accelerator=accelerator,
        devices=1,
        precision="16-mixed",
        gradient_clip_val=gradient_clip_value,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
