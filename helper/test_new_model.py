import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CLIP.clip as clip
from torchvision import transforms
import torch
from PIL import Image
import PIL
import numpy as np
import matplotlib.pyplot as plt

# torch.set_printoptions(threshold=torch.inf)
# np.set_printoptions(threshold=np.inf)

def preprocess_image(image_path, preprocess):
    target = Image.open(image_path)
    
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    transforms_ = []
    transforms_.append(transforms.Resize(224, interpolation=PIL.Image.BICUBIC))
    transforms_.append(transforms.CenterCrop(224))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)

    target = data_transforms(target).unsqueeze(0)

    normalize = transforms.Compose([preprocess.transforms[-1],])

    target = normalize(target).to("cuda")

    return target

def attn_map(image, model):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_weights.detach()
        print(cam)
        cams.append(cam)

    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49

    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance

model, preprocess = clip.load("ViT-B/32", device="cuda", jit = False)
model.eval().to("cuda")

# scripted_model = torch.jit.script(model)
# scripted_model.to("cuda")

image_input = preprocess_image("input_images/input_image.png", preprocess)

new_tensor = attn_map(image_input, model)

torch.save(new_tensor, "new_tensor.pt")
print(new_tensor)