from torchvision.transforms import v2


def augment(image, min_size: int = 1024) -> list:
    """Augments the PIL image and returns 10 variations of the input image."""
    W, H = image.size
    dim = max(min_size, min(int(H / 2), int(W / 2)))
    transform = v2.TenCrop((dim, dim))
    return list(transform(image))
