from predify import predify
from torchvision.models import VGG16_BN_Weights, vgg16_bn

weights = VGG16_BN_Weights.IMAGENET1K_V1
predify(vgg16_bn(weights=weights), './pvgg16_config.toml','./try_gen.py')
