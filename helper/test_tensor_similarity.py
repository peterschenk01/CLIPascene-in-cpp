import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import CLIP.clip as clip
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

cpp_tensor_model = torch.jit.load("cpp_tensor.pt")
cpp_tensor = list(cpp_tensor_model.parameters())[0]
cpp_tensor = cpp_tensor.to("cuda")

"""cpp_array = cpp_tensor.cpu().reshape(-1, tensor.size(-1)).numpy()
np.savetxt('cpp_tensor.txt', cpp_array, fmt='%.2f', delimiter=',')
python_array = tensor.cpu().reshape(-1, tensor.size(-1)).numpy()
np.savetxt('python_tensor.txt', python_array, fmt='%.2f', delimiter=',')"""

mse = torch.mean((tensor - cpp_tensor) ** 2).item()
print(f"Mean Squared Error: {mse}")
