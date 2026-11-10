import torch

from cnn_model import ASLCNN


# ---------------------------------------
# Device
# ---------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ---------------------------------------
# Create model
# ---------------------------------------

model = ASLCNN(
    num_classes=29
).to(device)


# ---------------------------------------
# Test input
# ---------------------------------------

dummy_input = torch.randn(
    4,
    3,
    64,
    64
).to(device)


# ---------------------------------------
# Forward pass
# ---------------------------------------

output = model(dummy_input)


# ---------------------------------------
# Print information
# ---------------------------------------

print(
    "Input shape:",
    dummy_input.shape
)

print(
    "Output shape:",
    output.shape
)

print(
    "Expected output shape:",
    "(4, 29)"
)

print("\nCNN model test successful!")