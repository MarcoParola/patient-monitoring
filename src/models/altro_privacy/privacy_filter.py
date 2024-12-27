import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BasePrivacyFilter(pl.LightningModule):
    """
    Base class for implementing privacy filters in video processing.
    Provides common functionality for training and evaluation.
    """
    def __init__(self, learning_rate=1e-4):
        """
        Initialize the base privacy filter.
        
        Args:
            learning_rate (float): Learning rate for the optimizer. Defaults to 1e-4.
        """
        super(BasePrivacyFilter, self).__init__()
        self.learning_rate = learning_rate

    def forward(self, x):
        """
        Forward pass of the model. Must be implemented by subclasses.
        
        Args:
            x (torch.Tensor): Input video tensor of shape (batch_size, channels, depth, height, width)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing:
                - Modified video with privacy filter applied
                - Intermediate features or filter mask
        """
        raise NotImplementedError("Forward method must be implemented in subclass.")

    def _common_step(self, batch, step_type):
        """
        Shared logic for training, validation, and test steps.
        
        Args:
            batch (tuple): Tuple containing (video_tensor, labels)
            step_type (str): Type of step ('train', 'val', 'test')
            
        Returns:
            dict: Dictionary containing loss and output tensors
        """
        video, _ = batch
        
        # Apply privacy filter
        output, filter_mask = self(video)
        
        # Calculate reconstruction loss between modified and original video
        loss = F.mse_loss(output, video)
        
        # Log metrics
        self.log(
            f"{step_type}_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True
        )
        
        return {
            'loss': loss,
            'output': output,
            'filter_mask': filter_mask
        }

    def training_step(self, batch, batch_idx):
        """Execute training step."""
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Execute validation step."""
        return self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        """Execute test step."""
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class BlurPrivacyFilter(BasePrivacyFilter):
    """
    Privacy filter that applies a 3D blurring effect to videos using depthwise convolution.
    """
    def __init__(self, channels, kernel_size=5, learning_rate=1e-4):
        """
        Initialize the blur privacy filter.
        
        Args:
            channels (int): Number of input video channels
            kernel_size (int): Size of the blur kernel. Defaults to 5
            learning_rate (float): Learning rate for optimization. Defaults to 1e-4
        """
        super(BlurPrivacyFilter, self).__init__(learning_rate)
        
        # Initialize 3D depthwise convolution for blurring
        self.blur_layer = nn.Conv3d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=channels,  # Use depthwise convolution for efficient processing
            bias=False
        )
        
        # Initialize blur kernel weights as uniform average filter
        with torch.no_grad():
            kernel_volume = kernel_size ** 3
            self.blur_layer.weight.fill_(1.0 / kernel_volume)

    def forward(self, x):
        """
        Apply blur filter to input video.
        
        Args:
            x (torch.Tensor): Input video tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Video with blur effect applied
                - Blur mask (intermediate feature map)
        """
        blur_mask = self.blur_layer(x)
        return x + blur_mask, blur_mask


class PixelatePrivacyFilter(BasePrivacyFilter):
    """
    Privacy filter that applies pixelation effect through downsampling and upsampling.
    """
    def __init__(self, kernel_size=2, learning_rate=1e-4):
        """
        Initialize the pixelation privacy filter.
        
        Args:
            kernel_size (int): Size of pooling kernel for downsampling. Defaults to 5
            learning_rate (float): Learning rate for optimization. Defaults to 1e-4
        """
        super(PixelatePrivacyFilter, self).__init__(learning_rate)
        self.kernel_size = kernel_size
        self.pool_layer = nn.AvgPool3d(kernel_size=kernel_size, stride=kernel_size)
        
        # Add a learnable scaling factor
        self.scale_factor = nn.Parameter(torch.ones(1))

    def forward(self, x):
        """
        Apply pixelation filter to input video.
        
        Args:
            x (torch.Tensor): Input video tensor
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Video with pixelation effect applied
                - Pixelation mask (downsampled-upsampled feature map)
        """
        # Downsample using average pooling
        downsampled = self.pool_layer(x)
        
        # Upsample to original size using nearest neighbor interpolation
        pixelation_mask = F.interpolate(
            downsampled,
            size=x.shape[2:],  # Match temporal and spatial dimensions
            mode="nearest"
        )
        
        # Apply learnable scaling to the pixelation effect
        pixelation_mask = self.scale_factor * pixelation_mask
        
        return x + pixelation_mask, pixelation_mask


def test_privacy_filters():
    """
    Test function to demonstrate usage of privacy filters.
    """
    # Define sample video input dimensions
    batch_size, channels, depth, height, width = 5, 3, 20, 256, 256
    input_shape = (batch_size, channels, depth, height, width)
    
    # Create random input video and dummy labels
    video_input = torch.randn(input_shape)
    labels = torch.zeros(batch_size)
    batch = (video_input, labels)

    # Test blur filter
    blur_filter = BlurPrivacyFilter(channels=channels)
    blur_output, blur_mask = blur_filter(video_input)
    print(f"Blur filter output shape: {blur_output.shape}")
    print(f"Blur mask shape: {blur_mask.shape}")

    # Test pixelation filter
    pixelate_filter = PixelatePrivacyFilter(kernel_size=5)
    pixelate_output, pixelate_mask = pixelate_filter(video_input)
    print(f"Pixelation filter output shape: {pixelate_output.shape}")
    print(f"Pixelation mask shape: {pixelate_mask.shape}")


if __name__ == "__main__":
    test_privacy_filters()
