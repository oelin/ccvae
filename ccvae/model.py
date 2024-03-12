#@markdown Implement the autoencoder model.

from typing import Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block.

    Example
    -------
    >>> module = ResidualBlock(
    ...     embedding_channels=256,
    ...     condition_channels=1024,
    ... )
    >>> x = torch.randn((1, 256, 32, 32))
    >>> c = torch.randn((1, 1024))
    >>> x = module(x, c)  # Shape: (1, 256, 32, 32).
    """

    def __init__(self, 
        embedding_channels: int,
        condition_channels: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_channels : int
            The number of embedding channels.
        condition_channels : int
            The number of condition channels.
        """

        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=embedding_channels,
            out_channels=embedding_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv_2 = nn.Conv2d(
            in_channels=embedding_channels,
            out_channels=embedding_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.group_norm_1 = nn.GroupNorm(
            num_groups=32,
            num_channels=embedding_channels,
        )

        self.group_norm_2 = nn.GroupNorm(
            num_groups=32,
            num_channels=embedding_channels,
        )

        self.linear = nn.Linear(
            in_features=condition_channels,
            out_features=embedding_channels * 4,
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, E, H, W).
        c : torch.Tensor
            The condition tensor (B, C).
        
        Returns
        -------
        x : torch.Tensor
            The output tensor (B, E, H, W).
        """

        transform = F.gelu(self.linear(c)[:, :, None, None])
        alpha_1, beta_1, alpha_2, beta_2 = transform.chunk(4, dim=-3)

        x = x + self.conv_1(alpha_1 * self.group_norm_1(x) + beta_1)
        x = x + self.conv_2(alpha_2 * self.group_norm_2(x) + beta_2)

        return x


class DownsampleBlock(nn.Module):
    """Downsample block.

    Example
    -------
    >>> module = DownsampleBlock(embedding_channels=256)
    >>> x = torch.randn((1, 256, 32, 32))
    >>> x = module(x)  # Shape: (1, 512, 16, 16).
    """

    def __init__(self, embedding_channels: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_channels : int
            The number of embedding channels.
        """

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=embedding_channels,
            out_channels=embedding_channels * 2,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, E, H, W).
        
        Returns
        -------
        x : torch.Tensor
            The output tensor (B, E*2, H/2, W/2).
        """

        return F.gelu(self.conv(x))


class UpsampleBlock(nn.Module):
    """Upsample block.

    Example
    -------
    >>> module = UpsampleBlock(embedding_channels=512)
    >>> x = torch.randn((1, 512, 16, 16))
    >>> x = module(x)  # Shape: (1, 256, 32, 32).
    """

    def __init__(self, embedding_channels: int) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_channels : int
            The number of embedding channels.
        """

        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=embedding_channels,
            out_channels=embedding_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module."""

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = F.gelu(self.conv(x))

        return x


class EncoderBlock(nn.Module):
    """Encoder block.

    Example
    -------
    >>> module = EncoderBlock(embedding_channels=256, condition_channels=1024)
    >>> x = torch.randn((1, 256, 32, 32))
    >>> c = torch.randn((1, 1024))
    >>> x = module(x, c)  # Shape: (1, 512, 16, 16).
    """

    def __init__(self, 
        embedding_channels: int,
        condition_channels: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_channels : int
            The number of embedding channels.
        condition_channels : int
            The number of condition channels.
        """

        super().__init__()

        self.residual_block = ResidualBlock(
            embedding_channels=embedding_channels,
            condition_channels=condition_channels,
        )

        self.downsample_block = DownsampleBlock(
            embedding_channels=embedding_channels,
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, E, H, W).
        c : torch.Tensor
            The condition tensor (B, C).
        
        Returns
        -------
        x : torch.Tensor
            The output tensor (B, E*2, H/2, W/2).
        """

        x = self.residual_block(x, c)
        x = self.downsample_block(x)

        return x


class DecoderBlock(nn.Module):
    """Decoder block.

    Example
    -------
    >>> module = DecoderBlock(embedding_channels=512, condition_channels=1024)
    >>> x = torch.randn((1, 512, 16, 16))
    >>> c = torch.randn((1, 1024))
    >>> x = module(x, c)  # Shape: (1, 256, 32, 32).
    """

    def __init__(self, 
        embedding_channels: int,
        condition_channels: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        embedding_channels : int
            The number of embedding channels.
        condition_channels : int
            The number of condition channels.
        """

        super().__init__()

        self.upsample_block = UpsampleBlock(
            embedding_channels=embedding_channels,
        )

        self.residual_block = ResidualBlock(
            embedding_channels=embedding_channels // 2,
            condition_channels=condition_channels,
        )
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, E, H, W).
        c : torch.Tensor
            The condition tensor (B, C).
        
        Returns
        -------
        x : torch.Tensor
            The output tensor (B, E/2, H*2, W*2).
        """

        x = self.upsample_block(x)
        x = self.residual_block(x, c)

        return x


@dataclass(frozen=True)
class GaussianDistribution:
    """Gaussian distribution."""

    mean: torch.Tensor
    variance: torch.Tensor

    def sample(self) -> torch.Tensor:
        """Sample from the distribution."""

        noise = torch.randn_like(self.mean)
        variance = torch.exp(-0.5 * self.variance)
        x = self.mean + variance*noise

        return x


@dataclass(frozen=True)
class AutoencoderConfiguration:
    """Autoencoder configuration."""

    input_channels: int
    latent_channels: int
    embedding_channels: int
    condition_channels: int
    layers: int


class Autoencoder(nn.Module):
    """Autoencoder.

    Example
    -------
    >>> configuration = AutoencoderConfiguration(
    ...     input_channels=3,
    ...     latent_channels=4,
    ...     embedding_channels=256,
    ...     condition_channels=1024,
    ...     layers=4,
    ... )
    >>> module = Autoencoder(configuration=configuration)
    >>> x = torch.randn((1, 3, 256, 256))
    >>> c = torch.randn((1, 1024))
    >>> x, posterior = module.encode(x, c)
    """

    def __init__(self, configuration: AutoencoderConfiguration) -> None:
        """Initialize the module.

        Parameters
        ----------
        configuration : AutoencoderConfiguration
            The module configuration.
        """

        super().__init__()

        # Input to embedding.
        
        self.conv_1 = nn.Conv2d(
            in_channels=configuration.input_channels,
            out_channels=configuration.embedding_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Embedding to latent.

        self.conv_2 = nn.Conv2d(
            in_channels=configuration.embedding_channels * (2 ** configuration.layers),
            out_channels=configuration.latent_channels * 2,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Latent to embedding.

        self.conv_3 = nn.Conv2d(
            in_channels=configuration.latent_channels,
            out_channels=configuration.embedding_channels * (2 ** configuration.layers),
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Embedding to input.

        self.conv_4 = nn.Conv2d(
            in_channels=configuration.embedding_channels,
            out_channels=configuration.input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        # Encoder blocks.

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                embedding_channels=configuration.embedding_channels * (2 ** i),
                condition_channels=configuration.condition_channels,
            ) for i in range(configuration.layers)
        ])

        # Decoder blocks.

        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                embedding_channels=configuration.embedding_channels * (2 ** i),
                condition_channels=configuration.condition_channels,
            ) for i in reversed(range(1, configuration.layers + 1))
        ])

        # Zero initialize conditioning parameters.
        # TODO
    
    def encode(self, x: torch.Tensor, c: torch.Tensor) -> GaussianDistribution:
        """Encode a tensor.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, E, H, W).
        c : torch.Tensor
            The condition tensor (B, C).
        
        Returns
        -------
        posterior : GaussianDistribution
            The posterior distribution.
        """

        x = self.conv_1(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, c)
        
        mean, variance = self.conv_2(x).chunk(2, dim=-3)

        posterior = GaussianDistribution(
            mean=mean,
            variance=variance,
        )

        return posterior
    
    def decode(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Decode a tensor.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, E, H, W).
        c : torch.Tensor (B, C).

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        """

        x = self.conv_3(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, c)
        
        x = F.sigmoid(self.conv_4(x))

        return x
    
    @torch.no_grad()
    def sample(
        self, 
        c: torch.Tensor, 
        prior: GaussianDistribution,
    ) -> torch.Tensor:
        """Sample from the model.

        Parameters
        ----------
        c : torch.Tensor
            The condition tensor (B, C).
        prior : GaussianDistribution.
            The prior distribution.
        
        Returns
        -------
        x : torch.Tensor
            The output tensor (B, E, H, W).
        """

        x = self.decode(prior.sample(), c)

        return x
    
    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor,
    ) -> Tuple[torch.Tensor, GaussianDistribution]:
        """Forward the module.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor (B, E, H, W).
        c : torch.Tensor (B, C).

        Returns
        -------
        x : torch.Tensor
            The output tensor.
        posterior: GaussianDistribution
            The posterior distribution.
        """

        posterior = self.encode(x, c)
        x = self.decode(posterior.sample(), c)

        return x, posterior


def vae_loss(
    reconstruction: torch.Tensor,
    posterior: GaussianDistribution,
    input: torch.Tensor,
    regularization_weight: float,
) -> torch.Tensor:
    """VAE loss."""

    reconstruction_loss = F.binary_cross_entropy(reconstruction, input)
    regularization_loss = (
        posterior.variance
        - posterior.mean.square()
        - posterior.variance.exp()
        + 1
    ).mean() * 0.5

    loss = reconstruction_loss - regularization_weight * regularization_loss

    return loss
