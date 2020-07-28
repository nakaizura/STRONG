'''Utils of STRL'''

import numpy as np
import os
import glob
import torch as th
from torchvision import models, transforms
from PIL import Image
from torch.autograd import Variable

#Dataloader
def dataloader(path):
    dataloader=np.load(path)
    data_all=list()
    for i in range(len(dataloader)):
        data = list()
        data.append(dataloader[i]['video_info'])
        data.append(dataloader[i]['w_vec'])
        data.append(dataloader[i]['v_feature'])
        data.append(dataloader[i]['w_start'])
        data.append(dataloader[i]['w_end'])
        data.append(dataloader[i]['fps'])

        data_all.append(data)

    return data_all

#Feature extraction
def extractor(img_path, net, use_gpu):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]
    )
    img = Image.open(img_path)
    img = transform(img)

    x = Variable(th.unsqueeze(img, dim=0).float(), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.data.numpy()

    return y.tolist()

#spp network
def extractor_spp(img_path, net, use_gpu,x,y,w,h):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),]
    )
    img = Image.open(img_path)
    #img_size
    width,high=480,270
    x=x*width
    y=y*high
    w=w*width
    h=h*high
    x=int(x-0.5*w)
    y=int(y-0.5*h)
    img=img.crop((x,y,x+w,y+h))
    img = transform(img)

    x = Variable(th.unsqueeze(img,dim=0), requires_grad=False)
    if use_gpu:
        x = x.cuda()
        net = net.cuda()
    y = net(x).cpu()
    y = y.data.numpy()

    return y.tolist()

#resnet
def resnet(path,start,end):
    files_list=list()
    for i in range(start,end):
        file_glob = os.path.join(path, str(i) + '.jpg')
        files_list.extend(glob.glob(file_glob))

    #resnet50
    resnet50_feature_extractor = models.resnet50(pretrained=False)
    resnet50_feature_extractor.load_state_dict(th.load('./resnet50-19c8e357.pth'))
    resnet50_feature_extractor.fc = th.nn.Linear(2048, 2048)
    th.nn.init.eye_(resnet50_feature_extractor.fc.weight)
    for param in resnet50_feature_extractor.parameters():
        param.requires_grad = False

    use_gpu = th.cuda.is_available()

    video_fearture=list()
    for x_path in [files_list[j] for j in range(len(files_list))]:
        try:
            sigle_feature=extractor(x_path, resnet50_feature_extractor, use_gpu)
        except: break
        video_fearture.append(sigle_feature)
    #feature=attention(video_fearture)
    feature=np.squeeze(np.array(video_fearture).mean(axis=0))
    return feature

#calculate temporal intersection over union
def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(abs(inter[1]-inter[0]))/(abs(union[1]-union[0]))
    return iou

#calculate TRL reward
def calculate_reward(Previou_IoU, current_IoU, t):
    penalty=0.01 #Step penalty factor 
    if current_IoU > Previou_IoU and Previou_IoU>=0:
        reward = 1-penalty*t
    elif current_IoU <= Previou_IoU and current_IoU>=0:
        reward = -penalty*t
    else:
        reward = -1-penalty*t
    return reward

#calculate SRL reward
def calculate_reward_img(Previou_Q, current_Q):
    if current_Q > Previou_Q:
        reward = 1
    elif current_Q == Previou_Q:
        reward = 0
    else:
        reward = -1
    return reward

def compute_IoU_recall_top_n(topn, IoU, iou_record):
    yes=0
    for i in range(len(iou_record)):
        if iou_record[i]>=IoU:
            yes=yes+1
    acc=yes/len(iou_record)

    return acc

#Log
def log(path,info):
    f = open(path, 'a')
    f.write(info)
    f.close()

#calculate mIoU
def compute_mIoU(iou_record):
    return np.mean(iou_record)
