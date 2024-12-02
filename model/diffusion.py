import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from model.block import DiffusionEmbedding, Encoder, ResidualBlock, Decoder, Conv1d_with_init


class diffusion_base(nn.Module):

    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )
        self.seqlen = config["seqlen"]

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)

        self.backward_projection = Conv1d_with_init(self.seqlen, self.seqlen, 1)
        self.cond_projection = Conv1d_with_init(self.channels + 1, self.channels, 1)

        self.encoder = Encoder(
            nn.ModuleList(
                [
                    ResidualBlock(
                        side_dim=config["side_dim"],
                        channels=self.channels,
                        diffusion_embedding_dim=config["diffusion_embedding_dim"],
                        nheads=config["nheads"],
                    )
                    for _ in range(config["layers"])
                ]
            )
        )
        self.decoder = Decoder(self.channels)

    def forward(
        self,
        x,
        cond_info,
        reverse_x,
        reverse_cond_info,
        negative_input,
        negative_cond_info,
        X_pred,
        diffusion_step,
    ):

        B, inputdim, K, L = x.shape

        x_hidden = self.__embedding(x, B, K, L, inputdim)
        reverse_x_hidden = self.__embedding(reverse_x, B, K, L, inputdim)
        negative_hidden = self.__embedding(negative_input, B, K, L, inputdim)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        forward_noise_hidden = self.encoder(x_hidden, cond_info, diffusion_emb)
        reverse_noise_hidden = self.encoder(
            reverse_x_hidden, reverse_cond_info, diffusion_emb
        )
        negative_hidden = self.encoder(
            negative_hidden, negative_cond_info, diffusion_emb
        )  # B,D,K*L

        if X_pred is not None:
            random_mask = torch.rand(B, 1, K, L).to(X_pred.device)
            pred_cond = (
                self.backward_projection(X_pred.permute(0, 2, 1))
                .permute(0, 2, 1)
                .unsqueeze(1)
            )  # B,1,K,L
            new_cond = pred_cond * random_mask + x * (1 - random_mask)
            new_cond = new_cond.reshape(B, 1, K * L)
            forward_noise_hidden = torch.cat([forward_noise_hidden, new_cond], dim=1)
            forward_noise_hidden = self.cond_projection(forward_noise_hidden)
        else:
            new_cond = x
            new_cond = new_cond.reshape(B, 1, K * L)
            forward_noise_hidden = torch.cat([forward_noise_hidden, new_cond], dim=1)
            forward_noise_hidden = self.cond_projection(forward_noise_hidden)

        forward_noise = self.decoder(forward_noise_hidden, B, K, L)

        return (
            forward_noise,
            forward_noise_hidden,
            reverse_noise_hidden,
            negative_hidden,
        )

    def impute(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x_enc = self.__embedding(x, B, K, L, inputdim)
        diffusion_emb = self.diffusion_embedding(diffusion_step)

        x_hidden = self.encoder(x_enc, cond_info, diffusion_emb)

        new_cond = x
        new_cond = new_cond.reshape(B, 1, K * L)
        x_hidden = torch.cat([x_hidden, new_cond], dim=1)
        x_hidden = self.cond_projection(x_hidden)

        x_noise = self.decoder(x_hidden, B, K, L)
        return x_noise

    # Private helper functions
    def __embedding(self, x, B, K, L, input_dim):
        if x is None:  # for pred_x in validation phase
            return None
        x = x.reshape(B, input_dim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        return x
