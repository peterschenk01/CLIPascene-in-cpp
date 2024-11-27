import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CLIP.clip as clip

# Downloads the CLIP model and converts it to TorchScript format

# Create the folder if it doesn't exist
os.makedirs(os.path.dirname("models/"), exist_ok=True)

model, preprocess = clip.load("ViT-B/32", device="cuda", jit = False)
model.eval()
scripted_model = torch.jit.script(model)
scripted_model.save("models/clip_vit_b32_scripted.pt")