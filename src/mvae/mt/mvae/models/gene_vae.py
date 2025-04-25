import torch
import torch.nn as nn
import torch.nn.functional as F

from src.mvae.mt.mvae.components import Component
from src.mvae.mt.mvae.models.vae import Reparametrized, ModelVAE
from torch import Tensor
from typing import List, Union


class GeneVAE(ModelVAE):

    def __init__(
        self, 
        n_gene_r: int,
        n_gene_p: int,
        n_batch: Union[int, List[int]],
        batch_invariant: bool,
        observation_dist: str,
        encoder_layer: List[int], 
        decoder_layer: List[int],
        components: List[Component], 
        scalar_parametrization: bool,
        beta: float = 1.0
    ) -> None:
        super().__init__(encoder_layer[-1], components, None, scalar_parametrization)
        self.n_gene_r = n_gene_r # Input genes
        self.n_gene_p = n_gene_p # Output genes
        if isinstance(n_batch, int):
            self.n_batch = [n_batch]
        else:
            self.n_batch = n_batch
        self.batch_invariant = batch_invariant
        self.observation_dist = observation_dist
        self.encoder_layer = encoder_layer
        self.decoder_layer = decoder_layer
        self.beta = beta

        self.x_dim = n_gene_p + sum(n_batch) if not batch_invariant else n_gene_p
        self.z_dim = sum(component.dim for component in components) + sum(n_batch)

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()
        self.fc_mu_logits = nn.Linear(self.decoder_layer[-1], self.n_gene_r)
        self.fc_sigma_logits = nn.Linear(self.decoder_layer[-1], self.n_gene_r)

    def _build_encoder(self):
        in_features = self.x_dim
        layers = []
        for units in self.encoder_layer:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ELU())
            in_features = units
        return nn.Sequential(*layers)
    
    def _build_decoder(self):
        in_features = self.z_dim
        layers = []
        for units in self.decoder_layer:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ELU())
            in_features = units
        return nn.Sequential(*layers)

    def _multi_one_hot(self, batch_idx: Tensor, depth_list: List[int]) -> Tensor:
        one_hots = []
        for i, depth in enumerate(depth_list):
            one_hots.append(F.one_hot(batch_idx[:, i], num_classes=depth))
        return torch.cat(one_hots, dim=1).float()
    
    def _prepare_inputs(self, x_r: Tensor, x_p: Tensor, batch_idx: Tensor) -> Tensor:
        library_size = x_r.sum(dim=-1, keepdim=True)
        batch = self._multi_one_hot(batch_idx, self.n_batch)
        return x_p, batch, library_size
    
    def _encode(self, x: Tensor, batch: Tensor) -> Tensor:
        if not self.batch_invariant:
            x = torch.cat([x, batch], dim=-1)
        x = self.encoder(x)
        return x
    
    def _decode(self, z: Tensor, batch: Tensor, library_size: Tensor) -> Tensor:
        z = torch.cat([z, batch], dim=-1)
        h = self.decoder(z)
        mu = F.softmax(self.fc_mu_logits(h), dim=-1) * library_size
        sigma = F.softplus(self.fc_sigma_logits(h)).mean(dim=0).expand_as(mu)
        return mu, sigma

    def log_likelihood(self, x: Tensor, mu: Tensor, sigma: Tensor, eps: float = 1e-10) -> Tensor:
        if self.observation_dist == "nb":
            log_mu_sigma = torch.log(mu + sigma + eps)
            ll = torch.lgamma(x + sigma) - torch.lgamma(sigma) - \
                torch.lgamma(x + 1) + sigma * torch.log(sigma + eps) - \
                sigma * log_mu_sigma + x * torch.log(mu + eps) - x * log_mu_sigma
            ll = torch.sum(ll, dim=-1)
        else:
            raise NotImplementedError()
        return ll

    def kl_divergence(self, reparametrized: List[Reparametrized]) -> Tensor:
        kl_components = []
        for component, r in zip(self.components, reparametrized):
            kl_ = component.kl_loss(r.q_z, r.p_z, r.z, r.data)
            assert torch.isfinite(kl_).all()
            kl_components.append(kl_)
        kl = torch.sum(torch.cat([x.unsqueeze(dim=-1) for x in kl_components], dim=-1), dim=-1)
        return kl

    def forward(self, x_r: Tensor, x_p: Tensor, batch_idx: Tensor) -> dict:
        x0, batch, library_size = self._prepare_inputs(x_r, x_p, batch_idx)
        h0 = self._encode(x0, batch)

        noise = torch.randn_like(x_p) * 0.1
        x1 = x0 + noise
        h1 = self._encode(x1, batch)
        reg = torch.sum((h0 - h1) ** 2, dim=-1) # Noise regularization

        reparametrized = []
        for component in self.components:
            q_z, p_z, _ = component(h0)
            z, data = q_z.rsample_with_parts()
            reparametrized.append(Reparametrized(q_z, p_z, z, data))

        z_concat = torch.cat([r.z for r in reparametrized], dim=-1)
        mu, sigma = self._decode(z_concat, batch, library_size)

        ll = self.log_likelihood(x_r, mu, sigma)
        kl = self.kl_divergence(reparametrized)
        elbo = ll - self.beta * kl

        return {
            "reparametrized": reparametrized,
            "latents": z_concat,
            "mu": mu,
            "sigma": sigma,
            "log_likelihood": ll,
            "kl_divergence": kl,
            "elbo": elbo,
            "reg": reg
        }