import torch
import torchvision.transforms as torch_transforms
import ser.transforms as transforms


def test_flip():
    tensor = torch.randn(1, 10, 10)

    # apply flip 
    flip_transform = transforms.flip()
    flipped_tensor = flip_transform(tensor)

    # check if shape remains the same
    assert tensor.shape == flipped_tensor.shape

    # check if flipped tensor is different from original tensor
    assert not torch.allclose(tensor, flipped_tensor)

    # check if flip was successful
    horizontal_flipped = torch_transforms.RandomHorizontalFlip(p=1.0)(tensor)
    fully_flipped = torch_transforms.RandomVerticalFlip(p=1.0)(horizontal_flipped)

    assert torch.allclose(fully_flipped, flipped_tensor)