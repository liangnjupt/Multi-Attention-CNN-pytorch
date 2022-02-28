import torch
from torchvision import models
import resnet
from torch import nn
from torch.nn import functional as F
import time
import numpy as np
import argparse
from logger import create_logger
import os
from datasets import build_dataloader
import datetime
from timm.utils import accuracy, AverageMeter
from utils import save_val_fig,save_checkpoint,build_optimizer,build_scheduler
import random
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import cv2
import vgg
from torch.utils.data import DataLoader,Dataset
from torch import optim as opt
from timm.scheduler.cosine_lr import CosineLRScheduler

parser = argparse.ArgumentParser()
# model
model_group = parser.add_argument_group(title='model options')
model_group.add_argument("--nparts", type=int, default=4, help="nparts")
model_group.add_argument("--alpha", type=float, default=0.01, help="weight of LocalMaxGlobalMin Loss")

# dataset
dataset_group = parser.add_argument_group(title='dataset options')
dataset_group.add_argument("--dataset_name", type=str, default="CUB", help="dataset name")
dataset_group.add_argument("--dataset_path", type=str, default="/home/liang/github/Dataset/bird_torch",
                           help="dataset path")
dataset_group.add_argument("--batch_size", type=int, default=8, help="batch size")
dataset_group.add_argument("--image_size", type=int, default=448, help="image_size")
dataset_group.add_argument("--nthreads", type=str, default=8, help="nthreads")

# train
train_group = parser.add_argument_group(title='train options')
train_group.add_argument("--train", action="store_true", help="train flag")
train_group.add_argument("--eval", action="store_true", help="eval flag")
train_group.add_argument("--epochs", type=int, default=100, help="epochs")
train_group.add_argument("--opt", type=str, default="SGD", choices=["SGD", "Adam"], help="optimizer type")
train_group.add_argument("--lr_scheduler", type=str, default="Step", choices=["Step", "Cosine"],
                         help="lr scheduler type")
train_group.add_argument("--lr", type=float, default=5e-4, choices=[5e-4, 1e-3], help="init learning rate")
train_group.add_argument("--warmup_lr", type=float, default=5e-7, help="warmup learning rate for cosine lr scheduler")
train_group.add_argument("--warmup_epochs", type=int, default=5, help="warmup epochs for cosine lr scheduler")
train_group.add_argument("--min_lr", type=float, default=5e-6, help="minimum learning rate for cosine lr scheduler")
train_group.add_argument("--eps", type=float, default=1e-8, help="eps")
train_group.add_argument("--betas", type=tuple, default=(0.9, 0.999), help="betas")
train_group.add_argument("--momentum", type=float, default=0.9, help="momentum")
train_group.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay")
train_group.add_argument("--print_freq", type=int, default=10, help="print frequency")
train_group.add_argument("--eval_freq", type=int, default=1, help="eval and save frequency")
train_group.add_argument('--output', default='MACNN_output', type=str, metavar='PATH',
                         help='root of output folder')
train_group.add_argument('--use_checkpoint', default=True, type=bool, help='use checkpoint')
args = parser.parse_args()



def preprocess(x):
    b, h, w = x.shape
    x = x.flatten(1)


    # x = x * 0.1
    # x = F.softmax(100*x, dim=1)
    # x = torch.exp(x)
    # x = x + 1
    # x = torch.log(x)
    # x = x * 4
    x = F.normalize(x, dim=-1, p=2).reshape((b, h, w))


    # 01 norm
    # x = 100 * (x - x.min(dim=1, keepdim=True)[0]) / (x.max(dim=1, keepdim=True)[0] - x.min(dim=1, keepdim=True)[0])
    #softmax
    # x = F.softmax(x, dim=-1).reshape((b, h, w))
    #normalize
    # x=F.normalize(x,dim=-1,p=2).reshape((b,h,w))
    x = x.reshape((b, h, w))
    return x

class DivLoss(nn.Module):
    def __init__(self):
        super(DivLoss, self).__init__()
        return
    def forward(self, x):
        '''
        :param x: [b,h,w]
        :return:
        '''

        x1=x[0]
        x2=x[1]
        x3=x[2]
        x4=x[3]

        # mgr = 0.
        # for data in [x1,x2,x3,x4]:
        #     mgr += data.mean()
        # mgr /= len(x)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        x4 = x4.view(x4.size(0), -1)
        tmp1 = torch.stack((x2, x3, x4))
        tmp2 = torch.stack((x1, x3, x4))
        tmp3 = torch.stack((x1, x2, x4))
        tmp4 = torch.stack((x1, x2, x3))
        t1, _ = torch.max(tmp1, dim=0)
        t2, _ = torch.max(tmp2, dim=0)
        t3, _ = torch.max(tmp3, dim=0)
        t4, _ = torch.max(tmp4, dim=0)
        # t1 = t1 - 0.01*mgr
        # t2 = t2 - 0.01*mgr
        # t3 = t3 - 0.01*mgr
        # t4 = t4 - 0.01*mgr
        loss = (torch.sum(x1 * t1) + \
                torch.sum(x2 * t2) + \
                torch.sum(x3 * t3) + \
                torch.sum(x4 * t4)) / x1.size(0)
        return loss


class DisLoss(nn.Module):
    def __init__(self):
        super(DisLoss, self).__init__()
        return

    def forward(self, x):
        '''
        :param x:  b,h,w
        :return:
        '''

        b,h,w=x.shape
        x = x.view(b, -1)
        num = torch.argmax(x, dim=1)
        cx = num % h
        cy = num // h
        maps = self.get_maps(h, cx, cy)
        maps = torch.from_numpy(maps).to(x.device)
        part = x * maps
        loss = torch.sum(part) / b
        return loss

    def get_maps(self, a, cx, cy):
        batch_size = len(cx)
        cx = cx.data.cpu().numpy()
        cy = cy.data.cpu().numpy()
        maps = np.zeros((batch_size, a * a), dtype=np.float32)
        rows = np.arange(a)
        cols = np.arange(a)
        coords = np.empty((len(rows), len(cols), 2), dtype=np.intp)
        coords[..., 0] = rows[:, None]
        coords[..., 1] = cols
        coords = coords.reshape(-1, 2)
        for b in range(batch_size):
            vec = np.array([cy[b], cx[b]])
            maps[b, :] = np.linalg.norm(coords - vec, axis=1)
        return maps

def zero_one_norm(featmaps):
    dimflag=False
    if featmaps.dim()==4:
        featmaps=featmaps.squeeze()
        dimflag=True
    b,h,w=featmaps.shape
    featmaps=featmaps.view(b,-1)
    featmaps=(featmaps-featmaps.min(dim=1,keepdim=True)[0])\
             /(featmaps.max(dim=1,keepdim=True)[0]-featmaps.min(dim=1,keepdim=True)[0])
    if dimflag:
        featmaps=featmaps.view(b,1,h,w)
    else:
        featmaps = featmaps.view(b, h, w)
    return featmaps

class SELayer(nn.Module):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.Tanh(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y=self.fc(y)#channel indicator
        y_ = y.view(b, c, 1, 1)

        attn_feat=x * y_.expand_as(x)

        M = torch.mean(attn_feat, dim=1, keepdims=True)
        M = zero_one_norm(M)
        return attn_feat,M.squeeze(),y


class MACNN(nn.Module):
    def __init__(self):
        super(MACNN,self).__init__()
        self.vgg=vgg.vgg19(True)
        # self.expansion = self.resnet.layer1[-1].expansion
        # self.feat_dims = self.expansion * 512
        self.feat_dims = 512
        self.se1 = SELayer(self.feat_dims)
        self.se2 = SELayer(self.feat_dims)
        self.se3 = SELayer(self.feat_dims)
        self.se4 = SELayer(self.feat_dims)

        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(self.feat_dims, 200)
        self.fc2 = nn.Linear(self.feat_dims, 200)
        self.fc3 = nn.Linear(self.feat_dims, 200)
        self.fc4 = nn.Linear(self.feat_dims, 200)
        self.fcnew=nn.Linear(4*self.feat_dims,200)

    def forward(self,x):
        feat_maps = self.vgg(x)
        f1, m1, l1 = self.se1(feat_maps)
        f2, m2, l2 = self.se2(feat_maps)
        f3, m3, l3 = self.se3(feat_maps)
        f4, m4, l4 = self.se4(feat_maps)

        pred1 = self.fc1(self.pool(f1).flatten(1))
        pred2 = self.fc2(self.pool(f2).flatten(1))
        pred3 = self.fc3(self.pool(f3).flatten(1))
        pred4 = self.fc4(self.pool(f4).flatten(1))
        pred=self.fcnew(self.pool(torch.cat([f1,f2,f3,f4],dim=1)).flatten(1))
        # pred = self.fcnew(self.pool(feat_maps).flatten(1))
        return feat_maps,[f1,f2,f3,f4],[m1,m2,m3,m4],[l1,l2,l3,l4],[pred1,pred2,pred3,pred4,pred]

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def getpos():
    indicators = np.zeros((512, len(dataset_train) * 2))
    for idx, (data, label) in enumerate(dataloader_train):
        # if idx>10:
        #     break
        print("getpos",idx)
        data = data.cuda()
        label = label.cuda()
        feat_maps, flist, chlist, fclist = model(data)
        B, C, H, W = feat_maps.shape
        for b in range(B):
            for c in range(C):
                m = feat_maps[b, c, :, :]
                argpos = m.argmax()
                argposx = argpos % H
                argposy = argpos // H
                # print(argposx,argposy)
                indicators[c, idx * B * 2 + b] = argposx
                indicators[c, idx * B * 2 + 1 + b] = argposy
    # print(indicators[0])
    return indicators

def clustering(indicators):
    cluster_pred = KMeans(n_clusters=4, random_state=0).fit_predict(indicators)
    indicators1 = list()
    indicators2 = list()
    indicators3 = list()
    indicators4 = list()
    for i in range(len(cluster_pred)):
        if cluster_pred[i] == 0:
            indicators1.append(i)
        elif cluster_pred[i] == 1:
            indicators2.append(i)
        elif cluster_pred[i] == 2:
            indicators3.append(i)
        elif cluster_pred[i] == 3:
            indicators4.append(i)

    # print(len(indicators1))
    # print(len(indicators2))
    # print(len(indicators3))
    # print(len(indicators4))
    # print(cluster_pred)
    return [indicators1,indicators2,indicators3,indicators4]


def vis(outputdir,indicators_list=None,draw_imgs=1):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    nums=0
    for idx, (data, label) in enumerate(dataloader_val):
        if nums>=draw_imgs:
            break
        nums += 1

        data = data.cuda()
        label = label.cuda()
        feat_maps, flist,mlist, chlist, fclist = model(data)

        B = data.size(0)
        data=data.cpu().detach()
        feat_maps=feat_maps.cpu().detach()

        for b in range(B):
            img=data[b]
            # inverse normalize
            img[0] = img[0] * std[0] + mean[0]
            img[1] = img[1] * std[1] + mean[1]
            img[2] = img[2] * std[2] + mean[2]
            img=img.permute(1,2,0).numpy()

            feat_maps_numpy=feat_maps[b].mean(0).numpy()
            feat_maps_numpy=cv2.resize(feat_maps_numpy,(448,448))

            # indicators_list
            t1 = np.zeros(512)
            t2 = np.zeros(512)
            t3 = np.zeros(512)
            t4 = np.zeros(512)
            t1[indicators_list[0]] = 1
            t2[indicators_list[1]] = 1
            t3[indicators_list[2]] = 1
            t4[indicators_list[3]] = 1

            # predicted indicators_list
            t1_ = chlist[0][b].cpu().detach().numpy()
            t2_ = chlist[1][b].cpu().detach().numpy()
            t3_ = chlist[2][b].cpu().detach().numpy()
            t4_ = chlist[3][b].cpu().detach().numpy()

            # 4 parts
            feat_maps1_numpy=feat_maps[b,indicators_list[0],:,:].mean(0).numpy()
            feat_maps1_numpy = cv2.resize(feat_maps1_numpy, (448, 448))
            feat_maps2_numpy = feat_maps[b, indicators_list[1], :, :].mean(0).numpy()
            feat_maps2_numpy = cv2.resize(feat_maps2_numpy, (448, 448))
            feat_maps3_numpy = feat_maps[b, indicators_list[2], :, :].mean(0).numpy()
            feat_maps3_numpy = cv2.resize(feat_maps3_numpy, (448, 448))
            feat_maps4_numpy = feat_maps[b, indicators_list[3], :, :].mean(0).numpy()
            feat_maps4_numpy = cv2.resize(feat_maps4_numpy, (448, 448))

            # predicted 4 parts
            feat_maps1_numpy_ = mlist[0][b].cpu().detach().numpy()
            feat_maps1_numpy_ = cv2.resize(feat_maps1_numpy_, (448, 448))
            feat_maps2_numpy_ = mlist[1][b].cpu().detach().numpy()
            feat_maps2_numpy_ = cv2.resize(feat_maps2_numpy_, (448, 448))
            feat_maps3_numpy_ = mlist[2][b].cpu().detach().numpy()
            feat_maps3_numpy_ = cv2.resize(feat_maps3_numpy_, (448, 448))
            feat_maps4_numpy_ = mlist[3][b].cpu().detach().numpy()
            feat_maps4_numpy_ = cv2.resize(feat_maps4_numpy_, (448, 448))


            plt.clf()
            plt.subplot(5,4,1)
            plt.imshow(img)
            plt.subplot(5,4,2)
            plt.imshow(feat_maps_numpy,cmap="jet")

            plt.subplot(5,4,5)
            plt.stem(np.arange(512), t1)
            plt.subplot(5,4,6)
            plt.stem(np.arange(512), t1)
            plt.subplot(5,4,7)
            plt.stem(np.arange(512), t3)
            plt.subplot(5,4,8)
            plt.stem(np.arange(512), t4)

            plt.subplot(5,4,9)
            plt.stem(np.arange(512), t1_)
            plt.subplot(5,4,10)
            plt.stem(np.arange(512), t2_)
            plt.subplot(5,4,11)
            plt.stem(np.arange(512), t3_)
            plt.subplot(5,4,12)
            plt.stem(np.arange(512), t4_)

            plt.subplot(5,4,13)
            plt.imshow(feat_maps1_numpy)
            plt.subplot(5,4,14)
            plt.imshow(feat_maps2_numpy)
            plt.subplot(5,4,15)
            plt.imshow(feat_maps3_numpy)
            plt.subplot(5,4,16)
            plt.imshow(feat_maps4_numpy)

            plt.subplot(5, 4, 17)
            plt.imshow(feat_maps1_numpy_)
            plt.subplot(5, 4, 18)
            plt.imshow(feat_maps2_numpy_)
            plt.subplot(5, 4, 19)
            plt.imshow(feat_maps3_numpy_)
            plt.subplot(5, 4, 20)
            plt.imshow(feat_maps4_numpy_)
            plt.show()


def freeze_model(freeze_modules="vgg"):
    '''
    :param freeze_modules: "vgg or se"
    :return:
    '''
    for k,v in model.named_parameters():
        if freeze_modules is not None:
            if freeze_modules in k:#vgg or se
                print("freeze:{}".format(k))
                v.requires_grad=False
            else:
                print("activate:{}".format(k))
                v.requires_grad=True
        else:
            print("activate:{}".format(k))
            v.requires_grad = True

@torch.enable_grad()
def train_step1(indicators_list):
    optimizer = opt.AdamW([{"params": model_withoutpl.se1.parameters()},
                        {"params": model_withoutpl.se2.parameters()},
                        {"params": model_withoutpl.se3.parameters()},
                        {"params": model_withoutpl.se4.parameters()}],
                          eps=args.eps,
                          betas=args.betas,
                          lr=0.001,
                          weight_decay=args.weight_decay)

    lr_scheduler=opt.lr_scheduler.MultiStepLR(optimizer,milestones=[2,4],gamma=0.1)
    inds1 = np.zeros(512)
    inds2 = np.zeros(512)
    inds3 = np.zeros(512)
    inds4 = np.zeros(512)
    inds1[indicators_list[0]] = 1
    inds2[indicators_list[1]] = 1
    inds3[indicators_list[2]] = 1
    inds4[indicators_list[3]] = 1
    inds1 = torch.from_numpy(inds1).view(1,512).float()
    inds2 = torch.from_numpy(inds2).view(1,512).float()
    inds3 = torch.from_numpy(inds3).view(1,512).float()
    inds4 = torch.from_numpy(inds4).view(1,512).float()

    criterion=nn.MSELoss()
    model.train()
    for epoch in range(5):
        for idx,datalabel in enumerate(dataloader_train):
            data=datalabel[0].cuda()
            label=datalabel[1].cuda()
            ind1 = inds1.repeat(label.shape[0],1).cuda()
            ind2 = inds2.repeat(label.shape[0],1).cuda()
            ind3 = inds3.repeat(label.shape[0],1).cuda()
            ind4 = inds4.repeat(label.shape[0],1).cuda()
            feat_maps, flist, mlist, chlist, fclist = model(data)
            loss=criterion(chlist[0], ind1)+criterion(chlist[1],ind2)+\
                 criterion(chlist[2], ind3)+criterion(chlist[3],ind4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx%10==0:
                print("epoch:{},idx:{},lr:{:.6f},loss:{:.8f}".format(epoch,idx,optimizer.param_groups[0]["lr"],loss.item()))
        lr_scheduler.step()
    torch.save(model_withoutpl.state_dict(), "MACNN.pkl")

@torch.enable_grad()
def train_step2():
    optimizer_se = opt.SGD([{"params": model_withoutpl.se1.parameters()},
                        {"params": model_withoutpl.se2.parameters()},
                        {"params": model_withoutpl.se3.parameters()},
                        {"params": model_withoutpl.se4.parameters()}],
                             # eps=args.eps,
                             # betas=args.betas,
                             lr=0.0001,
                             weight_decay=args.weight_decay)

    optimizer_cls = opt.SGD([{"params": model_withoutpl.vgg.parameters()},
                             {"params": model_withoutpl.fc1.parameters()},
                             {"params": model_withoutpl.fc2.parameters()},
                             {"params": model_withoutpl.fc3.parameters()},
                             {"params": model_withoutpl.fc4.parameters()},
                             {"params": model_withoutpl.fcnew.parameters()},
                             ],
                              lr=0.001,
                              weight_decay=args.weight_decay)

    criterion={"Dis":DisLoss(),
               "Div":DivLoss(),
               "Cls":nn.CrossEntropyLoss()}
    model.train()

    for epoch in range(20):
        for idx,datalabel in enumerate(dataloader_val):
            data=datalabel[0].cuda()
            label=datalabel[1].cuda()
            feat_maps, flist, mlist, chlist,fclist = model(data)
            if epoch % 2 != 0:
                # break
                # train se
                # if idx > 1:
                #     break
                optimizer_se.zero_grad()
                disloss = criterion["Dis"](mlist[0]) + criterion["Dis"](mlist[1]) + criterion["Dis"](mlist[2]) + \
                          criterion["Dis"](mlist[3])
                divloss = criterion["Div"](mlist)
                loss = divloss+disloss
                loss.backward()
                optimizer_se.step()
                if idx % 1 == 0:
                    print("train Se\tepoch:{},idx:{},disloss:{:.4f},divloss:{:.4f},loss:{:.4f}".format(epoch, idx,
                                                                                                       disloss.item(),
                                                                                                       divloss.item(),
                                                                                                       loss.item()))
            else:
                # if idx > 200:
                #     break
                # train cls
                optimizer_cls.zero_grad()
                clsloss = 1.0*(criterion["Cls"](fclist[0], label)+criterion["Cls"](fclist[1], label)+\
                          criterion["Cls"](fclist[2], label)+criterion["Cls"](fclist[3], label))\
                          +criterion["Cls"](fclist[4], label)

                loss = clsloss
                loss.backward()
                optimizer_cls.step()
                if idx % 10 == 0:
                    print("train Cls\tepoch:{},idx:{},loss:{:.4f}".format(epoch, idx, loss.item()))

def load_checkpoint(file="MACNN.pkl"):
    if args.use_checkpoint:
        state_dict=torch.load(file)
        model_state_dict=model_withoutpl.state_dict()
        new_state_dict={k:v for k,v in state_dict.items() if k in model_state_dict.keys()}
        for k,v in new_state_dict.items():
            print("resume:{}".format(k))
        model_state_dict.update(new_state_dict)
        model_withoutpl.load_state_dict(model_state_dict)

if __name__=="__main__":
    seed_torch()
    dataloader_train, dataloader_val, dataset_train, dataset_val, num_classes, n_iter_per_epoch_train = build_dataloader(
        args)
    model = MACNN()
    model.cuda()
    model_withoutpl = model

    for k,v in model.vgg.features[:28].named_parameters():
        v.requires_grad=False

    if torch.cuda.device_count()>1:
        model=torch.nn.DataParallel(model)

    load_checkpoint()

    # step1--get max pos
    # indicators=getpos()

    # np.save("indicators.npy",indicators)
    indicators=np.load("indicators.npy")

    # step2--hand-crafted clustering
    indicators_list=clustering(indicators)

    # step3--visualize
    # vis(os.path.join(args.output,"step3"),indicators_list)

    # step4--use indicator to train MACNN
    # train_step1(indicators_list)

    # step5--visualize
    # vis(os.path.join(args.output, "step4"), indicators_list)

    # step6--adjust MACNN
    train_step2()

    # step7--visualize
    vis(os.path.join(args.output, "step4"), indicators_list)

