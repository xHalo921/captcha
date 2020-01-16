import torch
import torch.nn as nn
from torchvision.transforms.functional import to_tensor
from torchvision.transforms import Compose, ToTensor
from datasets import CaptchaData
from models import CNN
from PIL import Image

source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
model_path = './model.pth'

cnn = CNN()
if torch.cuda.is_available():
    cnn = cnn.cuda()
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path))
else:
    cnn.eval()
    cnn.load_state_dict(torch.load(model_path, map_location='cpu'))

# img_path：单张图片路径
def captchaByPath(img_path):
    img = Image.open(img_path)
    img = to_tensor(img)
    if torch.cuda.is_available():
        img = img.view(1, 3, 32, 120).cuda()
    else:
        img = img.view(1, 3, 32, 120)
    output = cnn(img)
    output = output.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    output = output.view(-1, 4)[0]
    return ''.join([source[i] for i in output.cpu().numpy()])

# img_path：包含多张图片的目录
def captchaByDir(img_dir):
    transforms = Compose([ToTensor()])
    dataset = CaptchaData(img_dir, transform=transforms)
    lable = []

    for k, (img, target) in enumerate(dataset):
        if torch.cuda.is_available():
            img = img.view(1, 3, 32, 120).cuda()
        else:
            img = img.view(1, 3, 32, 120)
        output = cnn(img)
        output = output.view(-1, 36)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, 4)[0]
        lable.append(''.join([source[i] for i in output.cpu().numpy()]))
    return lable
