import numpy as np
import torch
import torch.nn as nn


class SDFDecoder(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_size=256, n_blocks=5, points_factor=1.0, **kwargs
    ):
        super().__init__()

        dims = [hidden_size] + [hidden_size for _ in range(n_blocks)] + [out_dim]
        self.num_layers = len(dims)

        for l in range(self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            setattr(self, "lin" + str(l), lin)

        self.fc_c = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size) for i in range(self.num_layers - 1)]
        )
        self.fc_p = nn.Linear(3, hidden_size)

        self.activation = nn.Softplus(beta=100)

        self.points_factor = points_factor

    def forward(self, points, point_feats):
        x = self.fc_p(points) * self.points_factor
        for l in range(self.num_layers - 1):
            x = x + self.fc_c[l](point_feats)
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x


class RGBDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim=3,
        hidden_size=256,
        n_blocks=5,
        points_factor=1.0,
        **kwargs
    ):
        super().__init__()

        dims = [hidden_size] + [hidden_size for _ in range(n_blocks)] + [out_dim]
        self.num_layers = len(dims)

        for l in range(self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            setattr(self, "lin" + str(l), lin)

        self.fc_p = nn.Linear(3, hidden_size)

        self.fc_c = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size) for i in range(self.num_layers - 1)]
        )
        self.activation = nn.ReLU()

        self.points_factor = points_factor

    def forward(self, points, point_feats):
        x = self.fc_p(points) * self.points_factor
        for l in range(self.num_layers - 1):
            x = x + self.fc_c[l](point_feats)
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        x = torch.sigmoid(x)
        return x


class SemanticDecoder(nn.Module):
    def __init__(
        self, in_dim, out_dim, hidden_size=256, n_blocks=5, points_factor=1.0, **kwargs
    ):
        super().__init__()

        dims = [hidden_size] + [hidden_size for _ in range(n_blocks)] + [out_dim]
        self.num_layers = len(dims)

        for l in range(self.num_layers - 1):
            lin = nn.Linear(dims[l], dims[l + 1])
            setattr(self, "lin" + str(l), lin)

        self.fc_p = nn.Linear(3, hidden_size)

        self.fc_c = nn.ModuleList(
            [nn.Linear(in_dim, hidden_size) for i in range(self.num_layers - 1)]
        )
        self.activation = nn.ReLU()

        self.points_factor = points_factor

    def forward(self, points, point_feats):
        x = self.fc_p(points) * self.points_factor
        for l in range(self.num_layers - 1):
            x = x + self.fc_c[l](point_feats)
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.activation(x)
        return x
