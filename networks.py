import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from listmodule import ListModule


class LREN(nn.Module):
    # Linearly Recurrent Encoder Network
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoder_layers = ListModule(self, 'encoder_')
        enc_shape = params['encoder_shape']
        self.dim = params['encoder_shape'][0]

        for i in range(len(enc_shape)-1):
            # bias flag
            bias = bool(i+2!=len(enc_shape))

            self.encoder_layers.append(nn.Linear(enc_shape[i],
                                                 enc_shape[i+1],
                                                 bias=bias))
        self.ko = nn.Linear(enc_shape[-1]+self.dim,
                            enc_shape[-1]+self.dim,
                            bias=False)

    def forward(self, x):
        # generate ground truth
        x = x[:, :self.params['n_shifts'], :]
        init_x = x.clone()
        for idx, layer in enumerate(self.encoder_layers):
            # relu activation on all encoder layers except for the last one
            if idx <= len(self.encoder_layers)-2:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        enc_gt = torch.cat((init_x, x), dim=-1)

        # generate trajectories from initial state
        enc_traj = enc_gt[:, 0:1, :].clone()
        for i in range(self.params['n_shifts']-1):
            enc_xi = self.ko(enc_traj[:, -1:, :])
            enc_traj = torch.cat((enc_traj, enc_xi), axis=1)
        return enc_gt, enc_traj


class DENIS(nn.Module):
    # Deep Encoder Network with Initial State parameterisation
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.encoder_layers = ListModule(self, 'encoder_')
        self.aux_layers = ListModule(self, 'ko_aux_')

        self.dim = params['encoder_shape'][0]
        self.ldim = params['aux_shape'][-1]

        enc_shape = params['encoder_shape']
        aux_shape = params['aux_shape'].copy()

        assert aux_shape[-1] == enc_shape[0] + enc_shape[-1]
        assert aux_shape[0] == enc_shape[0]

        aux_shape[-1] = aux_shape[-1]*aux_shape[-1]

        # initialize encoder network
        for i in range(len(enc_shape)-1):
            # bias flag (none for the last layer)
            bias = bool(i+2!=len(enc_shape))
            self.encoder_layers.append(nn.Linear(enc_shape[i],
                                                 enc_shape[i+1],
                                                 bias=bias))
        # initialize auxiliary network
        for j in range(len(aux_shape)-1):
            bias = bool(j+2!=len(aux_shape))
            self.aux_layers.append(nn.Linear(aux_shape[j],
                                             aux_shape[j+1],
                                             bias=bias))

    def forward(self, x):
        # Generate Koopman operator
        x0 = x[:, 0, :].clone()
        for idx, layer in enumerate(self.aux_layers):
            if idx <= len(self.aux_layers)-2:
                x0 = F.relu(layer(x0))
            else:
                x0 = layer(x0)
        # reshape to form Koopman matrix
        ko = x0.view((x0.shape[0], self.ldim, self.ldim))

        # generate ground truth
        x = x[:, :self.params['n_shifts'], :]
        init_x = x.clone()
        for idx, layer in enumerate(self.encoder_layers):
            # relu activation on all encoder layers except for the last one
            if idx <= len(self.encoder_layers)-2:
                x = F.relu(layer(x))
            else:
                x = layer(x)
        enc_gt = torch.cat((init_x, x), dim=-1)

        # generate trajectories from initial state
        enc_traj = enc_gt[:, 0:1, :].clone()
        for i in range(self.params['n_shifts']-1):
            enc_xi = torch.bmm(enc_traj[:, -1:, :], ko)
            enc_traj = torch.cat((enc_traj, enc_xi), axis=1)
        return enc_gt, enc_traj, ko


def koopman_loss(enc_gt, enc_traj, params):
    dim = params['encoder_shape'][0]
    state_error = enc_gt[:, :, :dim] - enc_traj[:, :, :dim]
    latent_error = enc_gt[:, :, dim:] - enc_traj[:, :, dim:]
    state_mse = torch.mean(state_error**2)
    latent_mse = torch.mean(latent_error**2)

    state_error_se = torch.sum(state_error, dim=2)
    state_inf_mse = torch.mean(state_error_se.norm(p=float('inf'), dim=1))

    loss = params["state_loss"] * state_mse + params["latent_loss"] * latent_mse + params["inf_loss"] * state_inf_mse
    return loss, state_mse, latent_mse, state_inf_mse
