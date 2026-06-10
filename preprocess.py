from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_style_norm(style_norm: str) -> str:
    if not style_norm:
        return "none"
    style_norm = style_norm.lower()
    if style_norm not in ("none", "imagenet"):
        raise ValueError(f"Unsupported style_norm: {style_norm}")
    return style_norm


def build_style_transform(image_size: int, style_norm: str = "none") -> transforms.Compose:
    style_norm = normalize_style_norm(style_norm)
    tfms = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if style_norm == "imagenet":
        tfms.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(tfms)


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
