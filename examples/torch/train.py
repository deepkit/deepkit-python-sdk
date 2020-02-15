import argparse
import base64
import io
import math
from struct import pack
from time import sleep

import PIL
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from deepkit.pytorch import get_pytorch_graph
from deepkit.utils import array_to_img
from deepkit.utils.image import get_image_tales, get_layer_vis_square
from examples.torch.resnet import ResNet18

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
net = net.to(device)
# if device == 'cuda':
#     net = torch.nn.DataParallel(net)
#     cudnn.benchmark = True

# l = [module for module in net.modules() if type(module) != nn.Sequential]

known_modules_map = dict()
known_modules_name_map = dict()

for name, module in net.named_modules():
    known_modules_map[module] = name
    known_modules_name_map[name] = module

import deepkit

x = trainset[0][0].unsqueeze(0)

context = deepkit.context(deepkit.ContextOptions(project='pytorch'))
context.set_model_graph(get_pytorch_graph(net, x))


def pil_image_to_jpeg(image):
    buffer = io.BytesIO()

    image.save(buffer, format="JPEG", optimize=True, quality=70)
    return buffer.getvalue()


def make_image_from_dense(neurons):
    cols = int(math.ceil(math.sqrt(len(neurons))))

    even_length = cols * cols
    diff = even_length - len(neurons)
    if diff > 0:
        neurons = np.append(neurons, np.zeros(diff, dtype=neurons.dtype))

    img = array_to_img(neurons.reshape((1, cols, cols)))
    img = img.resize((cols * 8, cols * 8))

    return img


def get_histogram(x, tensor):
    h = np.histogram(tensor.detach().numpy(), bins=20)
    # <version><x><bins><...x><...y>, little endian
    # uint8|Uint32|Uint16|...Float32|...Uint32
    # B|L|H|...f|...L
    return pack('<BIH', 1, int(x), h[0].size) + h[1].astype('<f').tobytes() + h[0].astype('<I').tobytes()


def register_hook(module):
    def hook(module, input, output):
        module_id = known_modules_map[module]
        print('executed', module_id)

        sample = output[0].detach().numpy()
        if len(sample.shape) == 3:
            image = PIL.Image.fromarray(get_image_tales(sample))
        elif len(sample.shape) > 1:
            if len(sample.shape) == 3:
                sample = np.transpose(sample, (2, 0, 1))
            image = PIL.Image.fromarray(get_layer_vis_square(sample))
        else:
            image = make_image_from_dense(sample)

        x = 1
        activations = get_histogram(x, output[0])
        whistogram = None
        bhistogram = None

        if hasattr(module, 'weight') and module.weight is not None:
            whistogram = get_histogram(x, module.weight)

        if hasattr(module, 'bias') and module.bias is not None:
            bhistogram = get_histogram(x, module.bias)

        context.client.job_action_threadsafe('addLiveLayerData', [
            module_id,
            base64.b64encode(pil_image_to_jpeg(image)).decode(),
            base64.b64encode(activations).decode(),
            base64.b64encode(whistogram).decode() if whistogram else None,
            base64.b64encode(bhistogram).decode() if bhistogram else None,
        ])

    module.register_forward_hook(hook)


net.apply(register_hook)

while True:
    net(trainset[0][0].unsqueeze(0))
    sleep(0.1)
