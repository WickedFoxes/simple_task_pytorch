from torchvision import transforms
from torchvision.transforms import InterpolationMode

STR2INTERP = {
    "nearest": InterpolationMode.NEAREST,
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "box": InterpolationMode.BOX,
    "hamming": InterpolationMode.HAMMING,
    "lanczos": InterpolationMode.LANCZOS,
}

# 필요 시 Albumentations 도 같은 방식으로 추가 등록 가능
_AUG_MAP = {
  "Resize": lambda **p: transforms.Resize(p["size"], interpolation=STR2INTERP[p.get("interpolation", "bilinear")]),
  "RandomCrop": lambda **p: transforms.RandomCrop(p["size"], padding=p.get("padding", 0)),
  "RandomResizedCrop": lambda **p: transforms.RandomResizedCrop(p["size"], scale=p.get("scale", (0.08, 1.0)), ratio=p.get("ratio", (3./4., 4./3.)), interpolation=STR2INTERP[p.get("interpolation", "bilinear")]),
  "RandomHorizontalFlip": lambda **p: transforms.RandomHorizontalFlip(p.get("p", 0.5)),
  "RandAugment": lambda **p: transforms.RandAugment(num_ops=p.get("n",2), magnitude=p.get("m",9)),
  "ToTensor": lambda **p: transforms.ToTensor(),
  "Normalize": lambda **p: transforms.Normalize(p["mean"], p["std"]),
  "RandomErasing": lambda **p: transforms.RandomErasing(
      p.get("p", 0.5), 
      scale=p.get("scale", (0.02, 0.33)),
      ratio=p.get("ratio", (0.3, 3.3)),
      value=p.get("value", 0),
      inplace=p.get("inplace", False)
  ),
}

def build_transform(pipes: list):
    ops = []
    for spec in pipes:
        name = spec.pop("name")
        ops.append(_AUG_MAP[name](**spec))
    return transforms.Compose(ops)