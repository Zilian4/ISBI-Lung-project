import torch
from monai.networks.nets import DenseNet121,EfficientNetBN,resnet
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis

# Define the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=2).to(device)
# model = EfficientNetBN(
#     model_name="efficientnet-b0",  # Options: "efficientnet-b0" to "efficientnet-b7"
#     spatial_dims=3,                # Use 3D convolutions for 3D medical imaging
#     in_channels=1,                 # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
#     num_classes=2                  # Number of output classes (e.g., 2 for binary classification)
# ).to(device)
# model = ShuffleNetV2().to(device)
model = MobileNetV2().to(device)
model = resnet.resnet50(n_input_channels=1).to(device)
# model = MobileNetV2(
#     spatial_dims=3,      # Use 3D convolutions for 3D medical imaging
#     in_channels=1,       # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
#     num_classes=2        # Number of output classes (e.g., 2 for binary classification)
# ).to(device)
# Create a dummy input for the model
# Adjust input dimensions based on your dataset, e.g., (batch_size, channels, depth, height, width)
dummy_input = torch.randn(1, 1, 64, 64, 64).to(device)

# Get parameter count
summary(model, input_size=(1, 64, 64, 64),device=str(device))

# Calculate FLOPs
flop_counter = FlopCountAnalysis(model, dummy_input)
flops = flop_counter.total()

print(f"FLOPs: {flops:.2e}")
