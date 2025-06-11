import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset




class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None, hsv=False, lab=False):
        self.img_labels = pd.read_csv(annotations_file)

      
        self.img_dir = img_dir
        self.transform = transform
        self.hsv = hsv
        self.lab = lab

        self.target_transform = target_transform
        self.num_category = len(self.img_labels.columns)-1#-1 is for the first column, which is image path not labels

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])     
        img_path = self.img_labels.iloc[idx, 0]   
        image = read_image(img_path).float()[0:3]
      
       
        labels = []
        for i in range(self.num_category):
            labels.append(self.img_labels.iloc[idx, 1+i])#+1 is for the first column, which is image path not labels
      
        label = torch.tensor(labels)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        if self.hsv:
            image = rgb_to_hsv(image)
        if self.lab:
            image = rgb_to_lab(image)
        return image, label, img_path

def rgb_to_hsv(image):
    # Assuming image is a tensor of shape (batch_size, 3, height, width) or (3, height, width)
    r, g, b = image[0], image[1], image[2]

    max_val, _ = torch.max(image, dim=0)
    min_val, _ = torch.min(image, dim=0)
    delta = max_val - min_val

    # Hue calculation
    hue = torch.zeros_like(max_val)
    
    mask = delta != 0
    r_eq_max = (r == max_val) & mask
    g_eq_max = (g == max_val) & mask
    b_eq_max = (b == max_val) & mask

    hue[r_eq_max] = (g[r_eq_max] - b[r_eq_max]) / delta[r_eq_max] % 6
    hue[g_eq_max] = ((b[g_eq_max] - r[g_eq_max]) / delta[g_eq_max]) + 2
    hue[b_eq_max] = ((r[b_eq_max] - g[b_eq_max]) / delta[b_eq_max]) + 4

    hue = hue / 6.0  # Normalize hue to [0, 1]
    hue[hue < 0] += 1  # Ensure hue is non-negative

    # Saturation calculation
    saturation = torch.zeros_like(max_val)
    saturation[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]

    # Value calculation
    value = max_val

    # Combine the HSV channels back
    hsv_image = torch.stack([hue, saturation, value], dim=0)

    return hsv_image

def rgb_to_lab(image):
    # Assumes image is a PyTorch tensor of shape [3, H, W] with values in [0, 1]

    # Step 1: RGB to Linear RGB
    mask = (image > 0.04045)  # This will now be a boolean tensor
    image = torch.where(mask, torch.pow((image + 0.055) / 1.055, 2.4), image / 12.92)

    # Step 2: Linear RGB to XYZ
    rgb_to_xyz = torch.tensor([[0.412453, 0.357580, 0.180423],
                               [0.212671, 0.715160, 0.072169],
                               [0.019334, 0.119193, 0.950227]]).to(image.device)
    
    image = image.permute(1, 2, 0)  # Change to [H, W, 3] for matrix multiplication
    image = torch.matmul(image, rgb_to_xyz.T)

    # Normalize for D65 illuminant
    image = image / torch.tensor([0.95047, 1.0, 1.08883]).to(image.device)

    # Step 3: XYZ to LAB
    epsilon = 0.008856
    kappa = 903.3

    mask = (image > epsilon)  # Boolean mask
    image = torch.where(mask, torch.pow(image, 1/3), (kappa * image + 16) / 116)

    # Calculate L, a, b
    L = 116 * image[..., 1] - 16
    a = 500 * (image[..., 0] - image[..., 1])
    b = 200 * (image[..., 1] - image[..., 2])

    lab = torch.stack([L, a, b], dim=-1)  # Shape: [H, W, 3]
    lab = lab.permute(2, 0, 1)  # Convert back to [3, H, W]

    return lab
