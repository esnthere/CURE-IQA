# -*- coding: utf-8 -*-

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
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

from model.simple_tokenizer import SimpleTokenizer as _Tokenizer
import pylab
import json




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
            image_features = image_features.unsqueeze(1).repeat(1, 11, 1).view(-1, 512)
            logits1 = torch.sum(logit_scale * image_features * text_features1, 1).unsqueeze(1)
            logits1 = logits1.view(-1, 11)
        else:
            logits1 = logits0 * 0

        return logits0, logits1, logits0 * 0


class DataLoader(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())



class Mydataset(Dataset):
    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = torch.FloatTensor(labels)

    def __getitem__(self, index):
        # img=self.imgs[index, :, 10:10 + 224, 10:10 + 224].transpose(1,2,0)
        img = self.imgs[index].transpose(1, 2, 0)
        img = tf.to_tensor(img)
        return img, self.labels[index]

    def __len__(self):
        return (self.imgs).shape[0]





def test(model, test_loader, epoch, device, all_test_loss):
    model.eval()
    test_loss = 0

    op0 = []
    op1 = []
    op2 = []
    tg = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data,  target = data.to(device),target.to(device)

            data[:, 0] -= 0.485
            data[:, 1] -= 0.456
            data[:, 2] -= 0.406
            data[:, 0] /= 0.229
            data[:, 1] /= 0.224
            data[:, 2] /= 0.225
            target -= 1
            target /= 4
            with autocast():
                output_0, output_1, output_2 = model(data,data)
                output2_0 = F.softmax(output_0[:, :2])


                loss = -torch.sum( target[:, 0] * torch.log(output2_0[:, 0]) + (1 - target[:, 0]) * torch.log(output2_0[:, 1])) /  output_0.shape[0]


            all_test_loss.append(loss)
            test_loss += loss
            tg = np.concatenate((tg, target[:, 0].cpu().numpy()))

            op0 = np.concatenate((op0, output2_0[:, 0].detach().cpu().numpy()))

    test_loss /= (batch_idx + 1)

    # print('Test ALL Pearson0:', pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="pearson"))
    # print('Test  ALL Spearman0:', pd.Series((op0[::1])).corr((pd.Series(tg[::1])), method="spearman"))
    #

    return op0,tg




def main():
    device = torch.device("cuda")

    classnames = [['high quality', 'low quality']]

    clip_model, _ = longclip.load("./checkpoints/longclip-B.pt", device='cpu')
    clip_model.float()

    print("Building custom CLIP")
    model = CustomCLIP(classnames, clip_model)
    model.load_state_dict(torch.load('koniq_rt25.pt'))
    model = model.to(device)


    batch_size = 1000
    num_workers_test = 0

    all_data = sio.loadmat('/home/sts/datadisk/Datasets/IQA_datasets/KonIQ-10k/Koniq_224.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)

    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    del all_data
    rt=0.25

    for i in range(0, 9):
        if i > 0:
            X = np.concatenate((X, Xtest), axis=0)
            Y = np.concatenate((Y, Ytest), axis=0)
            ind = np.arange(0, X.shape[0])
            np.random.seed(i)
            np.random.shuffle(ind)

            Xtest = X[ind[int(len(ind) * rt):]]
            Ytest = Y[ind[int(len(ind) * rt):]]
            X = X[ind[:int(len(ind) * rt)]]
            Y = Y[ind[:int(len(ind) * rt)]] #the model is trained on this splition

    test_dataset = Mydataset(Xtest,Ytest)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,pin_memory=True)
    print("KonIQ Test Results:")

    all_test_loss = []
    a, tg = test(model, test_loader, -1, device, all_test_loss)

    print('ALL Pearson:', pd.Series(a).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(a).corr(pd.Series(tg), method="spearman"))



# # #####################################
    all_data = sio.loadmat('/mnt/datadisk/Datasets/IQA_datasets/LIVEW/livew_224.mat')
    X = all_data['X']
    Y = all_data['Y'].transpose(1, 0)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest'].transpose(1, 0)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset=Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)
    print("Livew Test Results:")

    all_test_loss = []
    a, tg = test(model, test_loader, -1, device, all_test_loss)

    print('ALL Pearson:', pd.Series(a).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(a).corr(pd.Series(tg), method="spearman"))


    #######################################################

    all_data = sio.loadmat('/mnt/datadisk/Datasets/IQA_datasets/CID2013/cid_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = (Y + 0) / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = (Ytest + 0) / 25 + 1
    del all_data

    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,pin_memory=True)

    all_test_loss = []
    print("CID Test Results:")
    all_test_loss = []
    a, tg = test(model, test_loader, -1, device, all_test_loss)


    print('ALL Pearson:', pd.Series(a).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(a).corr(pd.Series(tg), method="spearman"))


    ########################################################
#
    all_data = sio.loadmat('/mnt/datadisk/Datasets/IQA_datasets/SPAQ/spaq_224.mat')

    X = all_data['X']
    Y = all_data['Y']
    Y = Y.reshape(Y.shape[1], 1)
    Y = Y / 25 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest.reshape(Ytest.shape[1], 1)
    Ytest = Ytest / 25 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,pin_memory=True)

    all_test_loss = []
    print("SPAQ Test Results:")
    a, tg = test(model, test_loader, -1, device, all_test_loss)

    print('ALL Pearson:', pd.Series(a).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(a).corr(pd.Series(tg), method="spearman"))

    #
#     #######################################################
#
    all_data = sio.loadmat('/mnt/datadisk/Datasets/IQA_datasets/RBID/rbid_224.mat')
    X = all_data['X']
    Y = all_data['Y']
    Y = Y.reshape(Y.shape[0], 1)
    Y = Y * 0.8 + 1
    Xtest = all_data['Xtest']
    Ytest = all_data['Ytest']
    Ytest = Ytest.reshape(Ytest.shape[0], 1)
    Ytest = Ytest * 0.8 + 1
    del all_data
    test_dataset = Mydataset(np.concatenate((X, Xtest), axis=0), np.concatenate((Y, Ytest), axis=0))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers_test,
                              pin_memory=True)

    all_test_loss = []
    print("RBID Test Results:")
    a, tg = test(model, test_loader, -1, device, all_test_loss)

    print('ALL Pearson:', pd.Series(a).corr(pd.Series(tg), method="pearson"))
    print('ALL Spearman:', pd.Series(a).corr(pd.Series(tg), method="spearman"))


if __name__ == '__main__':
    main()

