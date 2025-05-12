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


class PairedTransform:
    def __init__(self):
        pass

    def __call__(self, img1, img2):
        assert img1.size == img2.size, "Image sizes of img1 and img2 must be the same."
        # 1. Random crop
        if random.random() > 0.9:
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                img1, scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            img1 = transforms.functional.resized_crop(img1, i, j, h, w, (256, 256))
            img2 = transforms.functional.resized_crop(img2, i, j, h, w, (256, 256))

        # 2. Random horizontal flip
        if random.random() > 0.9:
            img1 = transforms.functional.hflip(img1)
            img2 = transforms.functional.hflip(img2)
        # 3. Random vertical flip
        if random.random() > 0.9:
            img1 = transforms.functional.vflip(img1)
            img2 = transforms.functional.vflip(img2)
        # 4. Random rotation
        if random.random() > 0.9:
            angle = transforms.RandomRotation.get_params([-15, 15])
            img1 = transforms.functional.rotate(img1, angle)
            img2 = transforms.functional.rotate(img2, angle)
        # 5. Random affine
        if random.random() > 0.9:
            affine_params = transforms.RandomAffine.get_params(
                degrees=(0, 0),
                translate=(0.05, 0.05),
                scale_ranges=(0.95, 1.05),
                shears=(0, 5),
                img_size=img1.size,
            )
            img1 = transforms.functional.affine(img1, *affine_params, fill=0)
            img2 = transforms.functional.affine(img2, *affine_params, fill=0)
        # 6. Random color jitter
        if random.random() > 0.8:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            img1 = transforms.functional.adjust_brightness(img1, brightness)
            img2 = transforms.functional.adjust_brightness(img2, brightness)
            img1 = transforms.functional.adjust_contrast(img1, contrast)
            img2 = transforms.functional.adjust_contrast(img2, contrast)
            img1 = transforms.functional.adjust_saturation(img1, saturation)
            img2 = transforms.functional.adjust_saturation(img2, saturation)
            img1 = transforms.functional.adjust_hue(img1, hue)
            img2 = transforms.functional.adjust_hue(img2, hue)
        # 7. Random grayscale
        if random.random() > 0.9:
            img1 = transforms.functional.rgb_to_grayscale(img1, num_output_channels=3)
            img2 = transforms.functional.rgb_to_grayscale(img2, num_output_channels=3)
        # 8. Random Gaussian blur
        if random.random() > 0.9:
            sigma = random.uniform(0.1, 2.0)
            img1 = transforms.functional.gaussian_blur(img1, kernel_size=3, sigma=sigma)
            img2 = transforms.functional.gaussian_blur(img2, kernel_size=3, sigma=sigma)
        # 9. Random sharpness
        if random.random() > 0.9:
            img1 = transforms.functional.adjust_sharpness(img1, sharpness_factor=2)
            img2 = transforms.functional.adjust_sharpness(img2, sharpness_factor=2)
        # 10. Random gamma
        if random.random() > 0.9:
            img1 = transforms.functional.adjust_gamma(img1, gamma=0.8)
            img2 = transforms.functional.adjust_gamma(img2, gamma=0.8)
        # 11. ToTensor
        img1 = transforms.functional.to_tensor(img1)
        img2 = transforms.functional.to_tensor(img2)
        # 12. Random noise
        if random.random() > 0.9:
            noise = 0.01 * torch.randn_like(img1)
            img1 = img1 + noise
            img2 = img2 + noise
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        return img1, img2


class TaskDataset(Dataset):
    def __init__(self, degraded_dir, clean_dir):
        self.samples = []
        for t in ["rain", "snow"]:
            for i in range(1, 1601):
                d = os.path.join(degraded_dir, f"{t}-{i}.png")
                c = os.path.join(clean_dir, f"{t}_clean-{i}.png")
                if os.path.exists(d) and os.path.exists(c):
                    self.samples.append((d, c, 0 if t == "rain" else 1))
                else:
                    print(f"Missing file: {d} or {c}")
        self.transform = PairedTransform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        d_path, c_path, prompt = self.samples[idx]
        d_img = Image.open(d_path).convert("RGB")
        c_img = Image.open(c_path).convert("RGB")
        if self.transform:
            d_img, c_img = self.transform(d_img, c_img)
        else:
            d_img = torch.from_numpy(np.array(d_img)).float().permute(2, 0, 1) / 255.0
            c_img = torch.from_numpy(np.array(c_img)).float().permute(2, 0, 1) / 255.0
        return d_img, c_img, prompt


class PromptIRLightning(pl.LightningModule):
    def __init__(self, lr_max, lr_min, T_max):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn = torch.nn.L1Loss()
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.T_max = T_max
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        degraded, clean, _ = batch
        restored = self.net(degraded)
        loss = self.loss_fn(restored, clean)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        degraded, clean, _ = batch
        restored = self.net(degraded)
        loss = self.loss_fn(restored, clean)
        self.log("val_loss", loss, prog_bar=True)
        mse = F.mse_loss(restored, clean)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        self.log("val_psnr", psnr, prog_bar=True)
        return {"val_loss": loss, "val_psnr": psnr}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr_max, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=self.lr_min
        )
        return [optimizer], [scheduler]

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        psnr = self.trainer.callback_metrics.get("val_psnr")
        if psnr is not None:
            psnr_value = psnr.item() if hasattr(psnr, "item") else float(psnr)
            filename = f"epoch_{epoch}_PSNR_{psnr_value:.4f}.pth"
            torch.save(self.net.state_dict(), os.path.join("checkpoints", filename))


def get_dataloaders(dataset_root, batch_size, val_split, num_workers):
    degraded_dir = os.path.join(dataset_root, "train", "degraded")
    clean_dir = os.path.join(dataset_root, "train", "clean")
    dataset = TaskDataset(degraded_dir, clean_dir)
    n_val = int(val_split * len(dataset))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


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
