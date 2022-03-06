import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import argparse
import os
from datasets import build_dataloader
import random
from matplotlib import pyplot as plt
import cv2
import vgg
from torch import optim as opt
from cluster.selfrepresentation import ElasticNetSubspaceClustering

parser = argparse.ArgumentParser()

# dataset
dataset_group = parser.add_argument_group(title='dataset options')
dataset_group.add_argument("--dataset_name", type=str, default="CUB", help="dataset name")
dataset_group.add_argument("--dataset_path", type=str, default="/home/liang/github/Dataset/bird_torch",
                           help="dataset path")
dataset_group.add_argument("--batch_size", type=int, default=16, help="batch size")
dataset_group.add_argument("--image_size", type=int, default=224, help="image_size")
dataset_group.add_argument("--nthreads", type=str, default=8, help="nthreads")
args = parser.parse_args()

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
        # maps=F.normalize(maps,dim=-1,p=2)
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
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        y_ = y.view(b, c, 1, 1)

        #old version
        # P = x * y_.expand_as(x)
        # M = torch.mean(P, dim=1, keepdims=True)
        # P = F.avg_pool2d(P, (h, w))  # b,c,1,1

        #paper version
        # M=torch.sum(x * y_.expand_as(x),dim=1,keepdim=True)#b,1,h,w
        # # M=torch.sigmoid(M)
        # M=M/M.sum(dim=(-2,-1),keepdim=True)#b,1,h,w
        # P=x*M.expand_as(x)#b,1,h,w
        # P=F.avg_pool2d(P,(h,w))#b,c,1,1

        #paper revised version
        M=torch.sum(x * y_.expand_as(x),dim=1,keepdim=True)#b,1,h,w
        M=F.normalize(M.view(b,-1),dim=-1,p=2).view(b,1,h,w)
        P=x*M.expand_as(x)#b,1,h,w
        P=F.avg_pool2d(P,(h,w))#b,c,1,1

        return P,M.squeeze(dim=1),y

class MACNN(nn.Module):
    def __init__(self):
        super(MACNN,self).__init__()
        self.vgg=vgg.vgg19(True)
        self.feat_dims = 512
        self.se1 = SELayer(self.feat_dims)
        self.se2 = SELayer(self.feat_dims)
        self.se3 = SELayer(self.feat_dims)
        self.se4 = SELayer(self.feat_dims)

        self.pool=nn.AdaptiveAvgPool2d(1)

        self.cnnfc=nn.Linear(self.feat_dims, 200)

        self.fc1 = nn.Linear(self.feat_dims, 200)
        self.fc2 = nn.Linear(self.feat_dims, 200)
        self.fc3 = nn.Linear(self.feat_dims, 200)
        self.fc4 = nn.Linear(self.feat_dims, 200)
        self.fcall=nn.Linear(5*self.feat_dims,200)

    def forward(self,x):
        feat_maps = self.vgg(x)

        cnn_pred=self.cnnfc(self.pool(feat_maps).flatten(1))

        P1, M1, y1 = self.se1(feat_maps.detach())
        P2, M2, y2 = self.se2(feat_maps.detach())
        P3, M3, y3 = self.se3(feat_maps.detach())
        P4, M4, y4 = self.se4(feat_maps.detach())


        pred1 = self.fc1(P1.flatten(1))
        pred2 = self.fc2(P2.flatten(1))
        pred3 = self.fc3(P3.flatten(1))
        pred4 = self.fc4(P4.flatten(1))
        P=torch.cat([P1,P2,P3,P4,self.pool(feat_maps)],dim=1)
        pred=self.fcall(P.flatten(1))

        return feat_maps,cnn_pred,\
               [P1,P2,P3,P4],\
               [M1,M2,M3,M4],\
               [y1,y2,y3,y4],\
               [pred1,pred2,pred3,pred4,pred]

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

def train_cnn():
    #step1
    optimizer = opt.SGD([{"params": model.vgg.parameters()},
                         {"params": model.cnnfc.parameters()}],
                          momentum=0.9,
                          lr=0.001,
                          weight_decay=5e-4)

    lr_scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(30):
        print("epoch:{}".format(epoch))
        train_loss = 0.0
        val_loss = 0.0
        num_corrects_train = 0
        num_corrects_val = 0

        model.train()
        for idx, datalabel in enumerate(dataloader_train):
            with torch.enable_grad():
                data = datalabel[0].cuda()
                label = datalabel[1].cuda()
                feat_maps, cnn_pred,Plist, Mlist, ylist, predlist = model(data)
                loss=criterion(cnn_pred,label)
                if idx % 10 == 0:
                    print("idx:{},loss:{:.4f}".format(idx, loss.item()))

                pred = cnn_pred.argmax(dim=1)
                num_corrects_train+=torch.eq(pred,label).float().sum().item()
                train_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print("train,train_loss:{},train_accuracy:{}".format(train_loss/len(dataloader_train),
                                                             num_corrects_train/len(dataset_train)))
        model.eval()
        for idx, datalabel in enumerate(dataloader_val):
            with torch.no_grad():
                data = datalabel[0].cuda()
                label = datalabel[1].cuda()
                feat_maps, cnn_pred, Plist, Mlist, ylist, predlist = model(data)
                loss = criterion(cnn_pred, label)
                if idx%10==0:
                    print("idx:{},loss:{:.4f}".format(idx,loss.item()))

                pred = cnn_pred.argmax(dim=1)
                num_corrects_val += torch.eq(pred, label).float().sum().item()
                val_loss += float(loss.item())
        print("val,val_loss:{},val_accuracy:{}".format(val_loss / len(dataloader_val),
                                                             num_corrects_val/ len(dataset_val)))
        lr_scheduler.step()

def clustering(indicators):
    cluster_pred  = ElasticNetSubspaceClustering(n_clusters=4, algorithm='lasso_lars', gamma=50).fit_predict(indicators)
    # cluster_pred = KMeans(n_clusters=4, random_state=0).fit_predict(indicators)
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
    print(len(indicators1),len(indicators2),len(indicators3),len(indicators4))
    return [indicators1,indicators2,indicators3,indicators4]

def getpos():
    indicators = np.zeros((512, len(dataset_train) * 2))
    for idx, (data, label) in enumerate(dataloader_train):
        print("getpos",idx)
        data = data.cuda()
        label = label.cuda()
        feat_maps, _, _, _, _, _ = model(data)
        B, C, H, W = feat_maps.shape
        for b in range(B):
            for c in range(C):
                m = feat_maps[b, c, :, :]
                argpos = m.argmax()
                argposx = argpos % H
                argposy = argpos // H
                indicators[c, idx * B * 2 + b] = argposx
                indicators[c, idx * B * 2 + 1 + b] = argposy
    return indicators

@torch.enable_grad()
def pretrain_attn():
    optimizer = opt.AdamW([{"params": model.se1.parameters()},
                        {"params": model.se2.parameters()},
                        {"params": model.se3.parameters()},
                        {"params": model.se4.parameters()}],
                          eps=1e-8,
                          betas=(0.9, 0.999),
                          lr=0.001,
                          weight_decay=5e-4)

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
    for epoch in range(1):
        for idx,datalabel in enumerate(dataloader_train):
            data=datalabel[0].cuda()
            label=datalabel[1].cuda()
            ind1 = inds1.repeat(label.shape[0],1).cuda()
            ind2 = inds2.repeat(label.shape[0],1).cuda()
            ind3 = inds3.repeat(label.shape[0],1).cuda()
            ind4 = inds4.repeat(label.shape[0],1).cuda()
            feat_maps, cnn_pred, Plist, Mlist, ylist, predlist=model(data)
            loss=criterion(ylist[0], ind1)+criterion(ylist[1],ind2)+\
                 criterion(ylist[2], ind3)+criterion(ylist[3],ind4)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx%10==0:
                print("epoch:{},idx:{},lr:{:.6f},loss:{:.8f}".format(epoch,idx,optimizer.param_groups[0]["lr"],loss.item()))
        lr_scheduler.step()

def vis(draw_imgs=1):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    nums=0
    for idx, (data, label) in enumerate(dataloader_val):
        if nums>=draw_imgs:
            break
        nums += 1

        data = data.cuda()
        label = label.cuda()
        feat_maps, _, _, mlist, chlist, _ = model(data)

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
def train_attnandcnn():
    optimizer = opt.SGD([
        {"params": model.vgg.parameters(),"lr":0.001},
                         {"params": model.se1.parameters()},
                         {"params": model.se2.parameters()},
                         {"params": model.se3.parameters()},
                         {"params": model.se4.parameters()},
                         {"params": model.fc1.parameters(),"lr":0.001},
                         {"params": model.fc2.parameters(),"lr":0.001},
                         {"params": model.fc3.parameters(),"lr":0.001},
                         {"params": model.fc4.parameters(),"lr":0.001},
                         {"params": model.fcall.parameters(),"lr":0.001},
                         ],
                        momentum=0.9,
                        lr=0.005,
                        weight_decay=5e-4)

    lr_scheduler = opt.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50], gamma=0.1)
    criterion ={"cls":nn.CrossEntropyLoss(),
                "div":DivLoss(),
                "dis":DisLoss()}

    for epoch in range(30):
        print("epoch:{}".format(epoch))
        train_loss = 0.0
        val_loss = 0.0
        num_corrects_train = 0
        num_corrects_val = 0

        model.train()
        for idx, datalabel in enumerate(dataloader_train):
            with torch.enable_grad():
                data = datalabel[0].cuda()
                label = datalabel[1].cuda()
                _, _, _, Mlist, _, predlist = model(data)

                clsloss = (criterion["cls"](predlist[0], label)+criterion["cls"](predlist[1], label)\
                          +criterion["cls"](predlist[2], label)+criterion["cls"](predlist[3], label)\
                          +criterion["cls"](predlist[4], label))/5
                divloss=criterion["div"](Mlist)
                disloss=criterion["dis"](Mlist[0])+criterion["dis"](Mlist[1])+criterion["dis"](Mlist[2])+criterion["dis"](Mlist[3])
                loss=20*divloss+disloss+clsloss
                if idx % 10 == 0:
                    print("idx:{},divloss:{:.4f},disloss:{:.4f},clsloss:{:.4f}".format(idx, divloss.item(),disloss.item(),clsloss.item()))

                pred = predlist[-1].argmax(dim=1)
                num_corrects_train += torch.eq(pred, label).float().sum().item()
                train_loss += float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                # print(model.se1.fc[0].weight.grad.max())
                optimizer.step()

        print("train,train_loss:{},train_accuracy:{}".format(train_loss / len(dataloader_train),
                                                             num_corrects_train / len(dataset_train)))

        model.eval()
        if epoch%5==0:
            for idx, datalabel in enumerate(dataloader_val):
                with torch.no_grad():
                    data = datalabel[0].cuda()
                    label = datalabel[1].cuda()
                    _, _, _, Mlist, _, predlist = model(data)

                    clsloss = (criterion["cls"](predlist[0], label) + criterion["cls"](predlist[1], label) \
                               + criterion["cls"](predlist[2], label) + criterion["cls"](predlist[3], label) \
                               + criterion["cls"](predlist[4], label)) / 5
                    divloss = criterion["div"](Mlist)
                    disloss = criterion["dis"](Mlist[0]) + criterion["dis"](Mlist[1]) + criterion["dis"](Mlist[2]) + \
                              criterion["dis"](Mlist[3])
                    loss = 20*divloss+disloss+clsloss
                    if idx % 10 == 0:
                        print("idx:{},divloss:{:.4f},disloss:{:.4f},clsloss:{:.4f}".format(idx, divloss.item(),disloss.item(),clsloss.item()))
    
                    pred = predlist[-1].argmax(dim=1)
                    num_corrects_val += torch.eq(pred, label).float().sum().item()
                    val_loss += float(loss.item())
            print("val,val_loss:{},val_accuracy:{}".format(train_loss / len(dataloader_val),
                                                           num_corrects_val / len(dataset_val)))
        lr_scheduler.step()


if __name__=="__main__":
    seed_torch()
    dataloader_train, dataloader_val, dataset_train, dataset_val, num_classes, n_iter_per_epoch_train = build_dataloader(
        args)
    model = MACNN()

    model.cuda()

    # step1 update model with Lcls
    train_cnn()
    torch.save(model.state_dict(),"MACNN2_output/cnn1.pkl")

    # step2--hand-crafted clustering
    state_dict=torch.load("MACNN2_output/cnn1.pkl")
    modelstate=model.state_dict()
    newstate={k:v for k,v in modelstate.items() if k in state_dict.keys()}
    modelstate.update(newstate)
    model.load_state_dict(modelstate)


    indicators=getpos()
    np.save("MACNN2_output/indicators1.npy",indicators)
    indicators = np.load("MACNN2_output/indicators1.npy")
    indicators_list = clustering(indicators)

    #step3 pretrain attention module
    pretrain_attn()
    torch.save(model.state_dict(), "MACNN2_output/pretrain_attn1.pkl")

    model.load_state_dict(torch.load("MACNN2_output/pretrain_attn1.pkl"))
    # vis()

    #step4 fine-tune attention module and CNN with Lcls and Lcng
    train_attnandcnn()

    torch.save(model.state_dict(), "MACNN2_output/attnandcnn1.pkl")
    model.load_state_dict(torch.load("MACNN2_output/attnandcnn1.pkl"))
    vis()
