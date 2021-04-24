import torch
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import json

from model.gram_efficientnet import GramEfficientNet

use_gpu = torch.cuda.is_available()

net_name = 'efficientnet-b0'

image_size = GramEfficientNet.get_image_size(net_name)
img = Image.open('img.jpg')
tfms = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                           transforms.ToTensor(),
                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
img = tfms(img).unsqueeze(0)
for i in range(10):
    img[i]=img[0]

print(img.size())
labels_map = json.load(open('labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

model_ft = GramEfficientNet.from_pretrained(net_name, load_fc=False)

if use_gpu:
    model_ft = model_ft.cuda()
    img = Variable(img.cuda())

outputs = model_ft(img)
preds = torch.topk(outputs, k=5).indices.squeeze(0).tolist()

print('-----')
for idx in preds:
    label = labels_map[idx]
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{:<75} ({:.2f}%)'.format(label, prob*100))