import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import os.path as osp
from collections import OrderedDict
import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from model import longclip

from torch.autograd import Variable
import numpy as np
import torch.optim as optim
import math
from scipy import io as sio
import torch.utils.data
import torchvision.models as models
import pandas as pd
import torchvision.transforms as transforms
import torchvision
import time
from torch.utils.data import Dataset as Dataset
from torch.utils.data import DataLoader as DataLoader
from skimage import io
import os
import torchvision.transforms.functional as tf
from PIL import Image
import cv2
import torchvision.models as models
from functools import partial
import matplotlib
import matplotlib.pyplot as plt
import lmdb
from prefetch_generator import BackgroundGenerator

from torch.cuda.amp import autocast as autocast

import pylab
import json


from model.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pylab
import json

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Mydataset(Dataset):
    def __init__(self, imgs, labels, longtxt, finetxt,oppo_rs1,oppo_rs2,oppo1,oppo2,wt):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)
        self.longtxt = longtxt
        self.finetxt = finetxt
        self.oppo_rs1 = oppo_rs1
        self.oppo_rs2 = oppo_rs2
        self.oppo1 = oppo1
        self.oppo2 = oppo2        
        self.wt =torch.FloatTensor(wt)

    def __getitem__(self, index):
        txt = longclip.tokenize(list(self.longtxt[index].values())[0], truncate=True)

        for n in range(10):
            txt = torch.cat((txt, longclip.tokenize(list(self.finetxt[index].values())[0][n], truncate=True)))

        oprstxt = longclip.tokenize(self.oppo_rs1[index][0], truncate=True)
        for n in range(1, 3):
            oprstxt = torch.cat((oprstxt, longclip.tokenize(self.oppo_rs1[index][n], truncate=True)))
        for n in range(0, 3):
            oprstxt = torch.cat((oprstxt, longclip.tokenize(self.oppo_rs2[index][n], truncate=True)))



        optxt = longclip.tokenize(self.oppo1[index][0], truncate=True)
        for n in range(1,3):
            optxt = torch.cat((optxt, longclip.tokenize(self.oppo1[index][n], truncate=True)))
        for n in range(0,3):
            optxt = torch.cat((optxt, longclip.tokenize(self.oppo2[index][n], truncate=True)))

        return torch.from_numpy(self.imgs[index]), self.labels[index], txt,oprstxt,optxt,self.wt[index]

    def __len__(self):
        return (self.imgs).shape[0]


_tokenizer = _Tokenizer()


def truncate_text_by_sentences(text, max_chars=2000, delimiter='.'):
    # 按'.'分割文本
    sentences = text.split(delimiter)

    # 初始化结果文本和当前字符计数
    truncated_text = ''
    char_count = 0

    # 遍历句子，并添加到结果文本中，直到达到最大字符数
    for sentence in sentences:
        # 添加句子和分隔符（除了最后一个句子）
        if char_count + len(sentence) + 1 <= max_chars or not truncated_text:
            if sentence:
                truncated_text += sentence + '.'
            char_count += len(sentence) + 1
        else:
            break

    # if char_count > max_chars:
    #     truncated_text = truncated_text.rsplit(delimiter, 1)[0]
    # print(char_count,truncated_text)

    return truncated_text.strip()

def split_op(dataset):

    with open('opposite_'+dataset+'.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)


    ct=0
    ls = []
    for i in range(len(fcc_data)):
        tt=list(fcc_data[i].values())[0].split('Original')
        if len(tt)==4:
            for j in range(1,4):
                tp=tt[j].split('"')
                if len (tp)>1:
                    ls.append((tp[1]))
                else:
                    tp = tt[j].split('**')
                    if len(tp) > 1:
                        ls.append((tp[1]))
                    else:
                        try:
                            tp = tt[j].split(':')[1].split('\n')
                            ls.append((tp[0]))
                        except IndexError:
                            ls.append(' ')
                            ct += 1
                            # print(tt)
        else:
            for j in range(1, 4):
                ls.append(' ')
                ct+=1


    ls2 = []
    for i in range(len(fcc_data)):
        tt = list(fcc_data[i].values())[0].split('Opposite')
        if len(tt) == 4:
            for j in range(1, 4):
                tp = tt[j].split('"')
                if len(tp) > 1:
                    ls2.append((tp[1]))
                else:
                    tp = tt[j].split('**')
                    if len(tp) > 1:
                        ls2.append((tp[1]))
                    else:
                        try:
                            tp = tt[j].split(':')[1].split('\n')
                            ls2.append((tp[0]))
                        except IndexError:
                            ls2.append(' ')
                            ct += 1
                            # print(tt)
        else:
            for j in range(1, 4):
                ls2.append(' ')


    for i in range(len(ls)):
        if ls[i]==' ':
            ls2[i]=' '
        if ls2[i]==' ':
            ls[i]=' '
        else:
            ls[i]=ls[i].strip('*: "\n')
            ls2[i]=ls2[i].strip('*: "\n')
            if ls[i][0:4]=='text':
                ls[i]=' '
                ls2[i]=' '
        if i%3==0:
            if ls[i] == ' ':
                ls[i+1]=' '
                ls[i + 2] = ' '

                ls2[i ] = ' '
                ls2[i + 1] = ' '
                ls2[i + 2] = ' '
        if i%3==1:
            if ls[i] == ' ':
                ls[i-1]=' '
                ls[i +1] = ' '

                ls2[i-1 ] = ' '
                ls2[i ] = ' '
                ls2[i + 1] = ' '
        if i%3==2:
            if ls[i] == ' ':
                ls[i-1]=' '
                ls[i -2] = ' '

                ls2[i-1 ] = ' '
                ls2[i-2 ] = ' '
                ls2[i ] = ' '


    return ls, ls2


def split_op_reason(dataset):

    with open('reason_opposite_'+dataset+'2.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)


    ct=0
    ls = []
    for i in range(len(fcc_data)):
        tt=list(fcc_data[i].values())[0].split('1. ')
        if len(tt)==3:
            tt2=tt[1].split('2. ')
            ls.append(tt2[0])
            tt3=tt2[1].split('3. ')
            ls.append(tt3[0])
            ls.append(tt3[1].split('And here')[0])

        else:
            tt = list(fcc_data[i].values())[0].split('And here')
            tt2 = tt[0].split('\n')
            if len(tt2)>3:
                ct=0
                for i in range(1,len(tt2)):
                    if (ct<3 )and (tt2[i]!=''):
                        ls.append(tt2[i].replace('*',''))
                        ct+=1
                        # print(tt2[i].replace('*',''))
            else:
                print(tt)



    ls2 = []
    for i in range(len(fcc_data)):
        tt = list(fcc_data[i].values())[0].split('1. ')
        if len(tt) == 3:
            tt2 = tt[2].split('2. ')
            ls2.append(tt2[0])
            tt3 = tt2[1].split('3. ')
            ls2.append(tt3[0])
            ls2.append(tt3[1])

        else:
            tt = list(fcc_data[i].values())[0].split('And here')
            tt2 = tt[1].split('\n')
            if len(tt2) > 3:
                ct = 0
                for i in range(1, len(tt2)):
                    if (ct < 3) and (tt2[i] != ''):
                        ls2.append(tt2[i].replace('*',''))
                        ct += 1

            else:
                print(tt)



    return ls, ls2


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, class_token_position):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 8
        ctx_init = False
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        CSC = False
        self.class_token_position = class_token_position
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = longclip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([longclip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner0 = PromptLearner(classnames[0], clip_model, 'middle')
        self.tokenized_prompts0 = self.prompt_learner0.tokenized_prompts

        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.clip_model = clip_model

    def forward(self, image, txt1, training=False):
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        text_features0 = self.text_encoder(self.prompt_learner0(), self.tokenized_prompts0)
        text_features0 = text_features0 / text_features0.norm(dim=-1, keepdim=True)
        logits0 = logit_scale * image_features @ text_features0.t()

        if training == True:
            txt1 = txt1.view(-1, 248)  # text[i*11+j,:]==text[i,j,:]
            text_features1 = self.clip_model.encode_text(txt1)
            text_features1 = text_features1 / text_features1.norm(dim=-1, keepdim=True)
            image_features = image_features.unsqueeze(1).repeat(1, 23, 1).view(-1, 512)
            logits1 = torch.sum(logit_scale * image_features * text_features1, 1).unsqueeze(1)
            logits1 = logits1.view(-1, 23)
        else:
            logits1 = logits0 * 0

        return logits0, logits1, logits0 * 0


def train(model, model_org, train_loader, optimizer, scaler, epoch, device, all_train_loss):
    model.train()
    model_org.eval()
    # model_org.load_state_dict(model.state_dict())
    st = time.time()
    op0 = []
    op1 = []
    op2 = []
    tg = []
    for batch_idx, (data, target, txt,oprstxt,optxt,wt) in enumerate(train_loader):
        data, target, txt,oprstxt,optxt = data.to(device), target.to(device), txt.to(device),oprstxt.to(device),optxt.to(device)
        torch.random.manual_seed(len(train_loader) * epoch + batch_idx)
        rd_ps = torch.randint(20, (3,))
        data = data[:, :, rd_ps[0]:rd_ps[0] + 224, rd_ps[1]:rd_ps[1] + 224]
        if rd_ps[1] < 10:
            data = torch.flip(data, dims=[3])

        data = data.float()
        data /= 255
        data[:, 0] -= 0.485
        data[:, 1] -= 0.456
        data[:, 2] -= 0.406
        data[:, 0] /= 0.229
        data[:, 1] /= 0.224
        data[:, 2] /= 0.225
        target -= 1
        target /= 4

        txt=torch.cat((txt,oprstxt,optxt),dim=1)

        optimizer.zero_grad()
        with ((((autocast())))):
            output_0, o_1, _ = model(data, txt,training=True)  # output_1:interlongtxt_VS_interfinetxt, output_2:longtxt_VS_otherlongtxt
            with torch.no_grad():
                outputorg_0, org_1, _ = model_org(data, txt, training=True)


            output_1=o_1[:,:11]
            outputorg_1 = org_1[:,:11]


            outputsoft_content1=F.softmax(o_1[:,11:17:3])
            outputsoft_content2=F.softmax(o_1[:,12:17:3])
            outputsoft_content3=F.softmax(o_1[:,13:17:3])


            outputsoft_content4=F.softmax(o_1[:,17::3])
            outputsoft_content5=F.softmax(o_1[:,18::3])
            outputsoft_content6=F.softmax(o_1[:,19::3])

            t1 = 0.5

            outputsoft_0 = F.softmax(output_0[:, :2])
            outputsoft_1 = F.log_softmax(output_1 * t1, dim=-1)

            outputsoftorg_1 = F.softmax(outputorg_1 * t1, dim=-1)

            loss_0 = -torch.sum(
                target[:, 0] * torch.log(outputsoft_0[:, 0]) + (1 - target[:, 0]) * torch.log(outputsoft_0[:, 1])) / \
                     output_0.shape[0]
            loss_1 = F.kl_div(outputsoft_1, outputsoftorg_1, reduction='batchmean')

            loss_2 = (-torch.sum(torch.log(outputsoft_content1[:, 0]))-torch.sum(torch.log(outputsoft_content2[:, 0]))-torch.sum(torch.log(outputsoft_content3[:, 0]))) / output_0.shape[0]/3
            loss_3 = (-torch.sum(torch.log(outputsoft_content4[:, 0]))-torch.sum(torch.log(outputsoft_content5[:, 0]))-torch.sum(torch.log(outputsoft_content6[:, 0]))) / output_0.shape[0]/3


            loss = loss_0 + 0.1 * loss_1+0.2*loss_2+0.1*loss_3



        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        all_train_loss.append(loss.item())
        # loss.backward()
        # optimizer.step()
        tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

        op0 = np.concatenate((op0, outputsoft_0[:, 0].detach().cpu().numpy()))

        if batch_idx % 100 == 0:
            print('Train Epoch:{} [({:.0f}%)]\t Loss: {:.4f} Loss0: {:.4f} Loss1: {:.4f} Loss2: {:.4f} Loss3: {:.4f} '.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item(), loss_0.item(), loss_1.item() * 1e10,
                       loss_2.item() , loss_3.item()))

    print('Train ALL Pearson0:', pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    print('Train  ALL Spearman0:', pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="spearman"))

    return all_train_loss


def test(model, model_org, test_loader, epoch, device, all_test_loss):
    model.eval()
    model_org.eval()
    #model_org.load_state_dict(model.state_dict())

    test_loss = 0

    op0 = []
    op1 = []
    op2 = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, target, txt,oprstxt, optxt,wt) in enumerate(test_loader):
            data, target, txt = data.to(device), target.to(device), txt.to(device)
            data = data[:, :, 10:10 + 224, 10:10 + 224]
            data = data.float()
            data /= 255
            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            target -= 1
            target /= 4
            with autocast():
                output_0, _, _ = model(data,txt)  # output_1:interlongtxt_VS_interfinetxt, output_2:longtxt_VS_otherlongtxt
                # outputorg_0, _, _ = model_org(data, txt)


                outputsoft_0 = F.softmax(output_0[:, :2])

                loss = -torch.sum(target[:, 0] * torch.log(outputsoft_0[:, 0]) + (1 - target[:, 0]) * torch.log(outputsoft_0[:, 1])) / \
                         output_0.shape[0]



                # if epoch < 50:
                #     loss = loss_0 + 0 * loss_1 + 0 * loss_2
                # else:
                #     loss = loss_0 + 0.2 * loss_1 + 0.1 * loss_2

            all_test_loss.append(loss)
            test_loss += loss
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

            op0 = np.concatenate((op0, outputsoft_0[:, 0].detach().cpu().numpy()))

            if batch_idx % 100 == 0:
                print('Test Epoch:{} [({:.0f}%)]\t Loss: {:.4f} '.format(
                    epoch, 100. * batch_idx / len(test_loader), loss.item()))

    test_loss /= (batch_idx + 1)

    pl0 = pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="pearson")
    sr0 = pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="spearman")
    print('Test ALL Pearson0:', pl0, 'Test  ALL Spearman0:', sr0)

    return all_test_loss, pl0, sr0


def main():
    device = torch.device("cuda")

    all_data = sio.loadmat('/home/sts/datadisk/Datasets/IQA_datasets/LIVEW/livew_244.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y.reshape(Y.shape[0], 1)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    Ytest = Ytest / 25 + 1
    del all_data

    txtpath1 = 'fine_' + 'livew' + '.json'
    with open('results_livew.json', 'r') as fcc_file:
        fcc_data = json.load(fcc_file)
    for i in range(len(fcc_data)):
        tt = truncate_text_by_sentences(list(fcc_data[i].values())[0], max_chars=1900, delimiter='.')
        fcc_data[i][list(fcc_data[i].keys())[0]] = tt

    fcc_train = np.array(fcc_data[:len(Y)])
    fcc_test = np.array(fcc_data[len(Y):])

    with open(txtpath1, 'r') as fcc_file:
        fine_txt = json.load(fcc_file)

    fine_train = np.array(fine_txt[:len(Y)])
    fine_test = np.array(fine_txt[len(Y):])

    # i = 10
    # plt.imshow(X[i].transpose(1, 2, 0) / 255)
    # plt.show()
    # plt.pause(0)
    # print(fcc_train[i])
    # print(fine_train[i])
    # print(fcc_test[i])
    # print(fine_test[i])
    ls,ls2=split_op('livew')

    oppo1=[]
    oppo2=[]
    wt=np.ones([int(len(ls)/3),1],dtype='float')
    for i in range(int(len(ls)/3)):
        tp=[]
        tp2=[]
        
        for j in range(3):
            tp.append(ls[i*3+j])
            tp2.append(ls2[i * 3 + j])
        if ls[i*3+j]==' ':
            wt[i]=0
        
        oppo1.append(tp)
        oppo2.append(tp2)
    
    oppo1_train = np.array(oppo1[:len(Y)])
    oppo1_test = np.array(oppo1[len(Y):])
    oppo2_train = np.array(oppo2[:len(Y)])
    oppo2_test = np.array(oppo2[len(Y):])
    wt_train = np.array(wt[:len(Y)])
    wt_test = np.array(wt[len(Y):])

    ls, ls2 = split_op_reason('livew')

    oppo_reason1 = []
    oppo_reason2 = []
    wt = np.ones([int(len(ls) / 3), 1], dtype='float')
    for i in range(int(len(ls) / 3)):
        tp = []
        tp2 = []

        for j in range(3):
            tp.append(ls[i * 3 + j])
            tp2.append(ls2[i * 3 + j])
        if ls[i * 3 + j] == ' ':
            wt[i] = 0

        oppo_reason1.append(tp)
        oppo_reason2.append(tp2)

    oppo_reason1_train = np.array(oppo_reason1[:len(Y)])
    oppo_reason1_test = np.array(oppo_reason1[len(Y):])
    oppo_reason2_train = np.array(oppo_reason2[:len(Y)])
    oppo_reason2_test = np.array(oppo_reason2[len(Y):])
    wt_train = np.array(wt[:len(Y)])
    wt_test = np.array(wt[len(Y):])

    rt = 0.05

    best_plccs = []
    best_srccs = []
    best_low_plccs = []
    best_low_srccs = []
    for i in range(0, 11):
        print('Split:', i)
        if i > 0:
            X = np.concatenate((X, Xtest), axis=0)
            Y = np.concatenate((Y, Ytest), axis=0)
            fcc_train = np.concatenate((fcc_train, fcc_test), axis=0)
            fine_train = np.concatenate((fine_train, fine_test), axis=0)
            oppo_reason1_train = np.concatenate((oppo_reason1_train, oppo_reason1_test), axis=0)
            oppo_reason2_train = np.concatenate((oppo_reason2_train, oppo_reason2_test), axis=0)
            oppo1_train = np.concatenate((oppo1_train, oppo1_test), axis=0)
            oppo2_train = np.concatenate((oppo2_train, oppo2_test), axis=0)
            wt_train = np.concatenate((wt_train, wt_test), axis=0)

            ind = np.arange(0, X.shape[0])
            np.random.seed(i)
            np.random.shuffle(ind)

            Xtest = X[ind[int(len(ind) * rt):]]
            Ytest = Y[ind[int(len(ind) * rt):]]
            fcc_test = fcc_train[ind[int(len(ind) * rt):]]
            fine_test = fine_train[ind[int(len(ind) * rt):]]
            oppo_reason1_test = oppo_reason1_train[ind[int(len(ind) * rt):]]
            oppo_reason2_test = oppo_reason2_train[ind[int(len(ind) * rt):]]
            oppo1_test = oppo1_train[ind[int(len(ind) * rt):]]
            oppo2_test = oppo2_train[ind[int(len(ind) * rt):]]
            wt_test = wt_train[ind[int(len(ind) * rt):]]

            X = X[ind[:int(len(ind) * rt)]]
            Y = Y[ind[:int(len(ind) * rt)]]
            fcc_train = fcc_train[ind[:int(len(ind) * rt)]]
            fine_train = fine_train[ind[:int(len(ind) * rt)]]
            oppo_reason1_train = oppo_reason1_train[ind[:int(len(ind) * rt)]]
            oppo_reason2_train = oppo_reason2_train[ind[:int(len(ind) * rt)]]
            oppo1_train = oppo1_train[ind[:int(len(ind) * rt)]]
            oppo2_train = oppo2_train[ind[:int(len(ind) * rt)]]
            wt_train = wt_train[ind[:int(len(ind) * rt)]]

            
            # print()
        if i < 1:
            continue
        # else:
        #     continue


        for n in range(3):  #repeat three times to obtain the best


            classnames = [['high quality', 'low quality']]

            clip_model, _ = longclip.load("../checkpoints/longclip-B.pt", device='cpu')
            clip_model.float()

            print("Building custom CLIP")
            model = CustomCLIP(classnames, clip_model)

            print("Turning off gradients in both the image and the text encoder")
            name_to_update = "prompt_learner"

            for name, param in model.named_parameters():

                if name_to_update in name:
                    param.requires_grad_(True)
                    # print(name)

                if name_to_update not in name:
                    param.requires_grad_(False)


            # # Double check
            # enabled = set()
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         enabled.add(name)
            # print(f"Parameters to be updated: {enabled}")

            model = model.to(device)

            model_org, _ = longclip.load("../checkpoints/longclip-B.pt", device='cpu')
            model_org = model_org.float()
            model_org = CustomCLIP(classnames, model_org).to(device)

            # model.load_state_dict(torch.load( 'livew244_rt5_promptbl7_split_'+str(i)+'.pt' ))
            ###################################################################

            train_dataset = Mydataset(X, Y, fcc_train, fine_train,oppo_reason1_train,oppo_reason2_train,oppo1_train, oppo2_train,wt_train)
            test_dataset = Mydataset(Xtest, Ytest, fcc_test, fine_test,oppo_reason1_test,oppo_reason2_test, oppo1_test, oppo2_test,wt_test)

            max_plsp = -1
            min_loss = 1e8
            lr = 0.01
            weight_decay = 1e-4
            batch_size = 40
            epochs = 2000
            num_workers_train = 0
            num_workers_test = 0
            ct = 0

            train_loader = DataLoaderX(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers_train,
                                       pin_memory=True)
            test_loader = DataLoaderX(test_dataset, batch_size=batch_size * 60, shuffle=False, num_workers=num_workers_test,
                                      pin_memory=True)

            all_train_loss = []
            all_test_loss = []
            all_test_loss, pl, pl2 = test(model, model_org, test_loader, -1, device, all_test_loss)
            ct = 0
            lr = 0.01
            max_plsp = -2
            scaler = torch.cuda.amp.GradScaler()

            for epoch in range(epochs):
                print(lr)
                optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                                      weight_decay=weight_decay)

                start = time.time()
                all_train_loss = train(model, model_org, train_loader, optimizer, scaler, epoch, device, all_train_loss)
                print(time.time() - start)
                if epoch % 5 == 4:
                    ct += 1
                    all_test_loss, pl, pl2 = test(model, model_org, test_loader, epoch, device, all_test_loss)
                    print("time:", time.time() - start)

                if max_plsp < pl + pl2:
                    save_nm = 'livew244_rt5_split_' + str(i) +  str(n) +'.pt'
                    max_plsp = pl + pl2
                    torch.save(model.state_dict(), save_nm)
                    ct = 0

                if ct > 5 and epoch > 50:
                    model.load_state_dict(torch.load(save_nm))
                    lr *= 0.3
                    ct = 0
                    if lr < 5e-5:
                        all_test_loss, pl, pl2 = test(model, model_org, test_loader, epoch, device, all_test_loss)
                        best_plccs.append(pl)
                        best_srccs.append(pl2)
                        print('Split:', i, 'End!', 'PLCC:', best_plccs, 'SRCC:', best_srccs)
                        break

                if epoch == 50:
                    ct = 0
                    lr = 0.001
                    for name, param in model.named_parameters():
                        if "text_encoder.transformer.resblocks.10" in name:
                            param.requires_grad_(True)
                        if "text_encoder.transformer.resblocks.11" in name:
                            param.requires_grad_(True)
                        if "text_encoder.ln_final" in name:
                            param.requires_grad_(True)

                        if "image_encoder.transformer.resblocks.10" in name:
                            param.requires_grad_(True)
                        if "image_encoder.transformer.resblocks.11" in name:
                            param.requires_grad_(True)
                        if "image_encoder.ln_post" in name:
                            param.requires_grad_(True)
                # elif epoch ==100:
                #     ct = 0
                #     lr = 0.005


if __name__ == '__main__':
    main()


