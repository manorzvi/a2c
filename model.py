from typing import Tuple
import numpy as np
from loguru import logger
import torch
import torch.nn as nn
from utils import np2torch_obs


class A2CModel(nn.Module):
    def __init__(self, shape_in: Tuple[int, int, int], n_action: int, args):
        super(A2CModel, self).__init__()

        h, w, c = shape_in

        paddings        = [(0, 0), (0, 0)]
        dilations       = [(1, 1), (1, 1)]
        strides         = [(4, 4), (2, 2)]
        kernels         = [(8, 8), (4, 4)]
        out_channels    = [16, 32]

        h_latent = self.calc_conv_output_shape(h, paddings[0][0], dilations[0][0], kernels[0][0], strides[0][0])
        w_latent = self.calc_conv_output_shape(w, paddings[0][1], dilations[0][1], kernels[0][1], strides[0][1])

        h_latent = self.calc_conv_output_shape(h_latent, paddings[1][0], dilations[1][0], kernels[1][0], strides[1][0])
        w_latent = self.calc_conv_output_shape(w_latent, paddings[1][1], dilations[1][1], kernels[1][1], strides[1][1])

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=c,  out_channels=out_channels[0], kernel_size=kernels[0], stride=strides[0],
                      padding=paddings[0], dilation=dilations[0]), nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=out_channels[1], kernel_size=kernels[1], stride=strides[1],
                      padding=paddings[1], dilation=dilations[1]), nn.ReLU()
        )

        self.latent = nn.Sequential(
            nn.Linear(out_channels[-1] * h_latent * w_latent, 256), nn.ReLU()
        )

        self.critic_head = nn.Linear(256, 1)

        self.actor_head  = nn.Sequential(
            nn.Linear(256, n_action),
            nn.Softmax(dim=1)
        )

    def forward_base(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        x = self.latent(x)
        return x

    def forward_actor(self, x):
        x = self.forward_base(x)
        x = self.actor_head(x)
        return x

    def forward_critic(self, x):
        x = self.forward_base(x)
        x = self.critic_head(x)
        return x

    def forward(self, x):
        x = self.forward_base(x)
        value = self.critic_head(x)
        policy = self.actor_head(x)
        return policy, value

    @staticmethod
    def calc_conv_output_shape(conv_input_shape, padding, dilation, kernel, stride):
        return int(np.floor((conv_input_shape + (2 * padding) - (dilation * (kernel-1)) - 1) / stride) + 1)

    def check(self, env, args):
        obs = env.reset()
        obs = np2torch_obs(obs, env.observation_space.low, env.observation_space.high).to(args.device)
        logger.debug(f"Input obs.size={obs.size()}, obs.dtype={obs.dtype}")
        try:
            with torch.no_grad():
                policy, value = self(obs)
                logger.debug(f"Output policy.size={policy.size()}, value.size={value.size()}")
        except Exception as e:
            raise e






