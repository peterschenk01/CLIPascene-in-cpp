import torch
import CLIP.clip as clip

model, _ = clip.load("ViT-B/32", device="cpu")
model.eval()

# Create example inputs
example_image = torch.randn(1, 3, 224, 224)
example_text = clip.tokenize(["a photo of a cat"])

# Trace the model
traced_model = torch.jit.trace(model, (example_image, example_text))

# Save the traced model
traced_model.save("clip_vit_b32.pt")