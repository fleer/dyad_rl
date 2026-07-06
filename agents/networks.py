import torch
import torch.nn as nn


class MLPNetwork(nn.Module):
    """Multi-layer perceptron for Q-value estimation from flattened discrete state."""

    def __init__(self, input_dim: int, num_actions: int, hidden_sizes: list[int]):
        """Initialize MLP Network.

        Builds a feed-forward MLP used to predict Q-values for discrete actions.

        Args:
            input_dim (int): Flattened observation dimension.
            num_actions (int): Number of discrete actions.
            hidden_sizes (list[int]): Hidden layer sizes.

        Returns:
            None: Network modules are initialized in place.
        """
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
        """Run Forward Pass.

        Computes action-value predictions for a batch of inputs.

        Args:
            x (torch.Tensor): Input tensor of shape ``(batch, input_dim)``.

        Returns:
            torch.Tensor: Q-value tensor of shape ``(batch, num_actions)``.
        """
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
        """Initialize CNN Network.

        Builds a convolutional Q-network for RGB observations.

        Args:
            input_channels (int): Number of image channels.
            input_height (int): Input image height.
            input_width (int): Input image width.
            num_actions (int): Number of discrete actions.
            conv_channels (list[int]): Output channels per convolutional layer.
            conv_kernels (list[int]): Kernel sizes per convolutional layer.
            conv_strides (list[int]): Strides per convolutional layer.
            fc_hidden (int): Hidden size of the first fully connected layer.

        Returns:
            None: Network modules are initialized in place.
        """
        super().__init__()
        conv_layers: list[nn.Module] = []
        in_ch = input_channels
        for out_ch, k, s in zip(conv_channels, conv_kernels, conv_strides):
            conv_layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s))
            conv_layers.append(nn.ReLU())
            # conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
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
        """Run Forward Pass.

        Extracts convolutional features and maps them to action values.

        Args:
            x (torch.Tensor): Input tensor of shape
                ``(batch, channels, height, width)``.

        Returns:
            torch.Tensor: Q-value tensor of shape ``(batch, num_actions)``.
        """
        features = self.conv(x)
        return self.fc(features)
