"""
Visualisations Grad-CAM++ pour les tâches de classification.
"""
import math
import torch
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np


def reshape_transform_vit(tensor, num_extra_tokens: int = 1):
    result = tensor[:, num_extra_tokens:, :]
    num_patches = result.size(1)
    h = w = int(math.sqrt(num_patches))
    result = result[:, :h*w, :].reshape(result.size(0), h, w, result.size(2))
    return result.permute(0, 3, 1, 2)


def unnormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1)


def gradcam_visualization(model, img_tensor: torch.Tensor, target_class: int, task_id: int, prompt_length: int = 10):
    input_tensor = img_tensor.unsqueeze(0)
    orig_img = unnormalize(img_tensor).numpy().transpose(1, 2, 0)
    model.eval()
    target_layer = model.vit.blocks[-1]
    num_extra = 1 + (prompt_length if getattr(model, 'use_prompts', False) else 0)
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], reshape_transform=lambda x: reshape_transform_vit(x, num_extra))
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    visualization = show_cam_on_image(orig_img, grayscale_cam, use_rgb=True)
    return visualization
