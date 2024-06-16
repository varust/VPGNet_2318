# coding=UTF-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
# from network_swinir1 import *
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

"""
modify fast DVD(vedio denoising) 
"""

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized.
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """

    def __init__(self, embedding_dim=256, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # torch.nn.init.uniform_(self.embeddings.weight, 0, 3)
        torch.nn.init.uniform_(self.embeddings.weight, -1.0/self.num_embeddings, 1.0/self.num_embeddings)
        # torch.nn.init.orthogonal_(self.embeddings.weight)

    def gram_loss(self, x, y):
        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)
        y = y.reshape(b, h * w, c)

        gmx = x.transpose(1, 2) @ x / (h * w)
        gmy = y.transpose(1, 2) @ y / (h * w)

        return (gmx - gmy).square().mean()

    def forward(self, x, target=None):
        B, C, H, W = x.shape
        encoding_indices = self.get_code_indices(x, target)
        quantized0 = self.quantize(encoding_indices)
        quantized = quantized0.view(B, H, W, C)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # weight, encoding_indices = self.get_code_indices(x)
        # quantized = self.quantize(weight, encoding_indices)

        if not self.training:
            return quantized, encoding_indices

        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        # print("??????????????????",loss)

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        return quantized, loss, encoding_indices

    def get_code_indices(self, flat_x, target=None):
        # flag = self.training
        flat_x = flat_x.permute(0, 2, 3, 1).contiguous()
        z_flattened = flat_x.view(-1, self.embedding_dim)
        # z_flattened = F.normalize(z_flattened, p=2, dim=1)
        weight = self.embeddings.weight
        # weight = F.normalize(weight, p=2, dim=1)
        flag = False
        if flag:
            # print(target.dtype)
            # raise ValueError("target type error! ")
            encoding_indices = target
        else:
            # compute L2 distance
            distances = (
                    torch.sum(z_flattened ** 2, dim=1, keepdim=True) +
                    torch.sum(weight ** 2, dim=1) -
                    2. * torch.matmul(z_flattened, weight.t())
            )  # [N, M]
            # dis, encoding_indices = distances.topk(k=10)
            # index = F.gumbel_softmax(distances, tau=1, hard=False)
            # encoding_indices = torch.argmin(index, dim=1)  # [N,]
            encoding_indices = torch.argmin(distances, dim=1)  # [N,]
            # weight = F.softmax(dis / 2, dim=1)
        return encoding_indices
        # return weight, encoding_indices

    # def quantize(self, weight, encoding_indices):
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        # b, k = weight.size()
        # self.embeddings(encoding_indices)
        # quantized = torch.stack(
        #     [torch.index_select(input=self.embeddings.weight, dim=0, index=encoding_indices[i, :]) for i in range(b)])
        # weight = weight.view(b, 1, k).contiguous()
        # quantized = torch.bmm(weight, quantized).view(b, -1).contiguous()
        # return quantized
        return self.embeddings(encoding_indices)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(0.1, inplace=True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        # self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

class CPEN(nn.Module):
    def __init__(self, n_feats=64, n_encoder_res=6):
        super(CPEN, self).__init__()
        # self.scale = scale
        # if scale == 2:
        E1 = [nn.Conv2d(64, n_feats, kernel_size=3, padding=1),
                  nn.LeakyReLU(0.1, True)]
        # elif scale == 1:
        #     E1 = [nn.Conv2d(96, n_feats, kernel_size=3, padding=1),
        #           nn.LeakyReLU(0.1, True)]
        # else:
        #     E1 = [nn.Conv2d(64, n_feats, kernel_size=3, padding=1),
        #           nn.LeakyReLU(0.1, True)]
        E2 = [
            ResBlock(
                default_conv, n_feats, kernel_size=3
            ) for _ in range(n_encoder_res)
        ]
        E3 = [
            nn.Conv2d(n_feats, n_feats * 2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 2, n_feats * 4, kernel_size=3, padding=1, stride=2),
            nn.LeakyReLU(0.1, True),
            # nn.AdaptiveAvgPool2d(4),
            # nn.MaxPool2d(2),
        ]
        E = E1 + E2 + E3
        self.E = nn.Sequential(
            *E
        )

        self.mlp = nn.Sequential(
            nn.Conv2d(n_feats * 4, n_feats * 4, 1, 1, 0),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feats * 4, n_feats * 4, 1, 1, 0),
            nn.LeakyReLU(0.1, True)
        )
        self.VQ = VectorQuantizer(n_feats * 4, 512, 0.25)
        # self.pixel_unshuffle = nn.PixelUnshuffle(4)
        # self.pixel_unshufflev2 = nn.PixelUnshuffle(2)

    def forward(self, x):
        # gt0 = self.pixel_unshuffle(gt)
        # if self.scale == 2:
        #     feat = self.pixel_unshufflev2(x)
        # elif self.scale == 1:
        #     feat = self.pixel_unshuffle(x)
        # else:
        #     feat = x
        # fea = x
        # x = torch.cat([feat, gt0], dim=1)
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        if self.training:
            fea2, loss, idx = self.VQ(fea1)
            return fea1, fea2, loss, idx
        else:
            fea2, idx = self.VQ(fea1)
            return fea2, fea1, idx

class Decoding(nn.Module):
    def __init__(self, feat=64, dim=48, scale=1):
        super(Decoding, self).__init__()
        self.upMode = 'bilinear'
        self.scale = scale
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=feat, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D1_ending = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=dim*8, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D2_ending = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64*2, out_channels=dim*4, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        # self.D3_ending = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64*2, kernel_size=3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_channels=64*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1),
        #                         # nn.ReLU()
        #                         )
        self.D4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        # self.D4_ending = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_channels=32, out_channels=dim, kernel_size=3, stride=1, padding=1),
        #                         # nn.ReLU()
        #                         )
        # self.D4_ending1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32*2, kernel_size=3, stride=1, padding=1),
        #                         nn.ReLU(),
        #                         nn.Conv2d(in_channels=32*2, out_channels=dim*2, kernel_size=3, stride=1, padding=1),
        #                         # nn.ReLU()
        #                         )
        self.D5 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.D6 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                # nn.ReLU()
                                )
        self.ending = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=31, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )

    def forward(self, E):
        ## decoding blocks
        D1 = self.D2(F.interpolate(self.D1(E), scale_factor=2, mode=self.upMode))
        D2 = self.D3(F.interpolate(D1, scale_factor=2, mode=self.upMode))
        D3 = self.D4(F.interpolate(D2, scale_factor=2, mode=self.upMode))
        if self.scale == 2:
            D4 = self.D5(F.interpolate(D3, scale_factor=2, mode=self.upMode))
            return self.ending(D4)
        elif self.scale == 3:
            D4 = self.D5(F.interpolate(D3, scale_factor=3, mode=self.upMode))
            return self.ending(D4)
        elif self.scale == 4:
            D4 = self.D5(F.interpolate(D3, scale_factor=2, mode=self.upMode))
            D5 = self.D6(F.interpolate(D4, scale_factor=2, mode=self.upMode))
            return self.ending(D5)
        else:
            D4 = self.D5(D3)
            D5 = self.D6(D4)
            return self.ending(D5)

class DGSMPIR(nn.Module):

    def __init__(self, channel0, factor, patch_size):
        super(DGSMPIR, self).__init__()
        # self.G = _3DT_Net(channel0=channel0, factor=factor, patch_size =patch_size)

        self.E_pre_gt = nn.Sequential(nn.Conv2d(in_channels=channel0, out_channels=64, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1))
        Res = [
            ResBlock(
                default_conv, 64, kernel_size=3
            ) for _ in range(8)
        ]
        self.E_pre_lr = nn.Sequential(nn.Conv2d(in_channels=34, out_channels=32, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(),
                                      nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                      *Res)
        self.E = CPEN(n_feats=64, n_encoder_res=6)
        self.D = Decoding(feat=64*4, dim=24)
        # self.BAM = BlancedAttention(in_planes=256)
        self.L1 = nn.L1Loss()
        # self.pool = nn.Sequential(nn.Conv2d(in_channels=256*2, out_channels=64, kernel_size=3, stride=1, padding=1),
        #                           nn.ReLU(),
        #                           nn.Conv2d(in_channels=64, out_channels=256, kernel_size=1, stride=1, padding=0),
        #                           nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        # self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        # self.delta_5 = Parameter(torch.ones(1), requires_grad=True)
        #
        # torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        # torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)

        self.reset_parameters()

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         nn.init.xavier_normal_(m.weight.data)
    #         nn.init.constant_(m.bias.data, 0.0)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size*4 - h % (self.window_size*4)) % (self.window_size*4)
        mod_pad_w = (self.window_size*4 - w % (self.window_size*4)) % (self.window_size*4)
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        _, _, h, w = x.size()
        return x

    def forward(self, x, rgb, gt):
        if self.training:
            _, _, H, W = gt.shape
            Bic = F.interpolate(x, scale_factor=8, mode='bicubic')
            F_rgb = self.E_pre_lr(torch.cat([rgb, Bic], dim=1))
            F_gt = self.E_pre_gt(gt)
            IPRS1, S1_IPR, vq_loss, vq_idx = self.E(F_rgb)  # IPRS1 [B, 256]
            GTIPRS, _, vq_loss1, vq_idx1 = self.E(F_gt)

            rec = self.D(GTIPRS)
            rec2 = self.D(IPRS1)
            # _, up8_1, up8, up4, up2, up0 = self.D(IPRS1)
            rec_loss = self.L1(rec, gt) + self.L1(rec2, gt) +self.L1(F_gt, F_rgb)
            # cat = torch.cat([IPRS1, S1_IPR], dim=1)
            # sr = self.G(x, rgb, [cat, self.pool(cat)])

            return rec2, [vq_loss + vq_loss1 + rec_loss, vq_idx1]
        else:
            Bic = F.interpolate(x, scale_factor=8, mode='bicubic')
            IPRS1, S1_IPR, vq_idx = self.E(self.E_pre_lr(torch.cat([rgb, Bic], dim=1)))
            rec2 = self.D(IPRS1)

            return rec2, vq_idx

    def flops(self):
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        # flops += self.upsample.flops()
        return flops
