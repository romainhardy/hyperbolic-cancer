import pytorch_lightning as pl
import torch

from src.mvae.mt.mvae.utils import parse_components
from src.mvae.mt.mvae.models.gene_vae import GeneVAE


class GeneModule(pl.LightningModule):
    def __init__(self, config):
        super(GeneModule, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.automatic_optimization = False
        options = config["lightning"]["model"]["options"]
        options["components"] = parse_components(config["lightning"]["model"]["components"], config["lightning"]["model"]["fixed_curvature"])
        self.model = GeneVAE(**options)
        
    def _ncurv_param_cond(self, key: str) -> bool:
        return "nradius" in key or "curvature" in key

    def _pcurv_param_cond(self, key: str) -> bool:
        return "pradius" in key
    
    def _curv_step_condition(self) -> bool:
        return (not self.config["lightning"]["model"]["fixed_curvature"]) and (self.current_epoch > 10)
    
    def _create_dummy_optimizer(self) -> torch.optim.Optimizer:
        dummy_param = torch.nn.Parameter(torch.zeros(1, device=self.device))
        dummy_param.requires_grad = False
        return torch.optim.SGD(params=[dummy_param], lr=1e-4)

    def configure_optimizers(self):
        net_params = [
            v for key, v in self.model.named_parameters()
            if not self._ncurv_param_cond(key) and not self._pcurv_param_cond(key)
        ]
        neg_curv_params = [v for key, v in self.model.named_parameters() if self._ncurv_param_cond(key)]
        pos_curv_params = [v for key, v in self.model.named_parameters() if self._pcurv_param_cond(key)]
        self._neg_curv_flag = True if neg_curv_params else False
        self._pos_curv_flag = True if pos_curv_params else False
        
        net_optimizer = getattr(torch.optim, self.config["optimizer"]["class"])(net_params, **self.config["optimizer"]["options"])
        ncurv_optimizer = torch.optim.SGD(neg_curv_params, lr=1e-4) if self._neg_curv_flag else self._create_dummy_optimizer()
        pcurv_optimizer = torch.optim.SGD(pos_curv_params, lr=1e-4) if self._pos_curv_flag else self._create_dummy_optimizer()
        
        optimizers = [net_optimizer, ncurv_optimizer, pcurv_optimizer]

        net_scheduler = getattr(torch.optim.lr_scheduler, self.config["scheduler"]["class"])(
            net_optimizer,
            max_lr=self.config["optimizer"]["options"]["lr"],
            total_steps=self.trainer.estimated_stepping_batches,
            **self.config["scheduler"]["options"],
        )
        
        return optimizers, [net_scheduler]
    
    def training_step(self, batch):
        x, batch_idx = batch
        outputs = self.model.forward(x, batch_idx)
        elbo = outputs["elbo"].mean()
        reg = outputs["reg"].mean()
        kl = outputs["kl_divergence"].mean()
        ll = outputs["log_likelihood"].mean()
        loss = -elbo + reg

        optimizers = self.optimizers()
        net_optimizer, ncurv_optimizer, pcurv_optimizer = optimizers
        net_scheduler = self.lr_schedulers()

        for opt in optimizers:
            opt.zero_grad()

        self.manual_backward(loss)

        net_optimizer.step()
        net_scheduler.step()
        
        if self._neg_curv_flag and self._curv_step_condition():
            ncurv_optimizer.step()
        if self._pos_curv_flag and self._curv_step_condition():
            pcurv_optimizer.step()

        self.log("train/elbo", elbo, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/reg", reg, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/kl", kl, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/ll", ll, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, sync_dist=True)

        for param_group in net_optimizer.param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False)
        
        return {"loss": loss}
    
    def validation_step(self, batch):
        x_r, x_p, batch_idx = batch
        outputs = self.model.forward(x_r, x_p, batch_idx)
        elbo = outputs["elbo"].mean()
        reg = outputs["reg"].mean()
        kl = outputs["kl_divergence"].mean()
        ll = outputs["log_likelihood"].mean()
        loss = -elbo + reg

        self.log("valid/elbo", elbo, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/reg", reg, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/kl", kl, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/ll", ll, on_step=False, on_epoch=True, sync_dist=True)
        self.log("valid/loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        return {"loss": loss}