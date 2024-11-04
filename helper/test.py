import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CLIP_.clip as clip
from torchvision import transforms
import torch
from PIL import Image
import PIL
import numpy as np

def get_target(image_path):
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

    target_ = data_transforms(target).unsqueeze(0).to("cuda")
    return target_


model, preprocess = clip.load("ViT-B/32", device="cpu", jit = False)

model.eval().to("cuda")

target = get_target("input_images/input_image.png")

data_transforms = transforms.Compose([
                    preprocess.transforms[-1],
                ])

tensor = data_transforms(target).to("cuda")

#cpp_tensor_model = torch.jit.load("cpp_tensor.pt")
#cpp_tensor = list(cpp_tensor_model.parameters())[0]
#cpp_tensor = cpp_tensor.to("cuda")

#mse = torch.mean((tensor - cpp_tensor) ** 2).item()
#print(f"Mean Squared Error: {mse}")


def interpret(image, texts, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    print(image_attn_blocks[0].attn_probs)

    """num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = [] # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach() #attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1) # mean of the 12 something
        cams.append(cam)  
        R = R + torch.bmm(cam, R)
              
    cams_avg = torch.cat(cams) # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:] # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance"""

interpret(tensor, "", model, "cuda")