import torch
import torch.nn as nn


class MLPNetwork(nn.Module):
    """Multi-layer perceptron for Q-value estimation from flattened discrete state."""

    def __init__(self, input_dim: int, num_actions: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNNetwork(nn.Module):
    """Convolutional network for Q-value estimation from RGB pixel observations."""

    def __init__(
        self,
        input_channels: int,
        input_height: int,
        input_width: int,
        num_actions: int,
        conv_channels: list[int],
        conv_kernels: list[int],
        conv_strides: list[int],
        fc_hidden: int,
    ):
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch, k, s in zip(conv_channels, conv_kernels, conv_strides):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s))
            conv_layers.append(nn.ReLU())
            in_ch = out_ch
        conv_layers.append(nn.Flatten())
        self.conv = nn.Sequential(*conv_layers)

        # Compute flattened size by doing a forward pass with a dummy tensor
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            flat_size = self.conv(dummy).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        return self.fc(features)
