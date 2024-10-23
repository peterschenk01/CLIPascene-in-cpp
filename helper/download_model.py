import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CLIP.clip as clip

# Downloads the CLIP model and converts it for LibTorch
# For this you need the CLIP library. You can get it at https://github.com/openai/CLIP or as submodule of this Repository

# Load the CLIP model (ViT-B/32)
model, preprocess = clip.load("ViT-B/32", device="cpu")

# Set the model to evaluation mode
model.eval()

# Create dummy input tensors
# Image: batch size of 1, 3 channels, 224x224 size
example_image_input = torch.randn(1, 3, 224, 224)

# Text: Tokenized text input, example with 1 sentence
example_text_input = clip.tokenize(["This is a dummy text input."]).to("cpu")  # Shape: (1, 77)

# Trace the full model with both image and text inputs
traced_model = torch.jit.trace(model, (example_image_input, example_text_input))

# Create the folder if it doesn't exist
os.makedirs(os.path.dirname("models/"), exist_ok=True)

# Save the traced model to TorchScript format
traced_model.save("models/clip_vit_b32.pt")