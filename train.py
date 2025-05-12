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
