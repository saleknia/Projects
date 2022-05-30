import copy
import logging
import math
import torch
import torch.nn as nn
import numpy as np
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import torch.nn.functional as F

class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2)) ----> hidden = (out_channels=in_channels)
        x = x.flatten(2) # (B, hidden=Ci, n_patches)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden=Ci)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class Attention_org(nn.Module):
    def __init__(self, config, vis,channel_num, M_return=False):
        super(Attention_org, self).__init__()
        self.M_return = M_return
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.query3 = nn.ModuleList()
        self.query4 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0], channel_num[0], bias=False) # W query 1 T1 ---> C1 * C1 ---> 64 * 64
            query2 = nn.Linear(channel_num[1], channel_num[1], bias=False) # W query 2 T2 ---> C2 * C2 ---> 128 * 128
            query3 = nn.Linear(channel_num[2], channel_num[2], bias=False) # W query 3 T3 ---> C3 * C3 ---> 256 * 256
            query4 = nn.Linear(channel_num[3], channel_num[3], bias=False) # W query 4 T4 ---> C4 * C4 ---> 512 * 512
            key = nn.Linear( self.KV_size,  self.KV_size, bias=False) # W Key Concat(T1,T2,T3,T4) ---> C_sigma * C_sigma ---> 940 * 940
            value = nn.Linear(self.KV_size,  self.KV_size, bias=False) # W Value Concat(T1,T2,T3,T4) ---> C_sigma * C_sigma ---> 940 * 940
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.query3.append(copy.deepcopy(query3))
            self.query4.append(copy.deepcopy(query4))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False) # W Out ---> Ci * Ci
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False) # W Out ---> Ci * Ci
        self.out3 = nn.Linear(channel_num[2], channel_num[2], bias=False) # W Out ---> Ci * Ci
        self.out4 = nn.Linear(channel_num[3], channel_num[3], bias=False) # W Out ---> Ci * Ci
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])



    def forward(self, emb1,emb2,emb3,emb4, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_Q3_list = []
        multi_head_Q4_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        if emb1 is not None:
            for query1 in self.query1:
                Q1 = query1(emb1) # (B, n_patch, hidden=Ci)
                multi_head_Q1_list.append(Q1)
        if emb2 is not None:
            for query2 in self.query2:
                Q2 = query2(emb2) # (B, n_patch, hidden=Ci)
                multi_head_Q2_list.append(Q2)
        if emb3 is not None:
            for query3 in self.query3:
                Q3 = query3(emb3) # (B, n_patch, hidden=Ci)
                multi_head_Q3_list.append(Q3)
        if emb4 is not None:
            for query4 in self.query4:
                Q4 = query4(emb4) # (B, n_patch, hidden=Ci)
                multi_head_Q4_list.append(Q4)
        for key in self.key:
            K = key(emb_all) # (B, n_patch, C_sigma)
            multi_head_K_list.append(K)
        for value in self.value:
            V = value(emb_all) # (B, n_patch, C_sigma)
            multi_head_V_list.append(V)
        # print(len(multi_head_Q4_list))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None # (B, Head, n_patch, hidden=Ci)
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None # (B, Head, n_patch, hidden=Ci)
        multi_head_Q3 = torch.stack(multi_head_Q3_list, dim=1) if emb3 is not None else None # (B, Head, n_patch, hidden=Ci)
        multi_head_Q4 = torch.stack(multi_head_Q4_list, dim=1) if emb4 is not None else None # (B, Head, n_patch, hidden=Ci)
        multi_head_K = torch.stack(multi_head_K_list, dim=1) # (B, Head, n_patch, C_sigma)
        multi_head_V = torch.stack(multi_head_V_list, dim=1) # (B, Head, n_patch, C_sigma)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None # (B, Head, hidden=Ci, n_patch)
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None # (B, Head, hidden=Ci, n_patch)
        multi_head_Q3 = multi_head_Q3.transpose(-1, -2) if emb3 is not None else None # (B, Head, hidden=Ci, n_patch)
        multi_head_Q4 = multi_head_Q4.transpose(-1, -2) if emb4 is not None else None # (B, Head, hidden=Ci, n_patch)

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_scores3 = torch.matmul(multi_head_Q3, multi_head_K) if emb3 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_scores4 = torch.matmul(multi_head_Q4, multi_head_K) if emb4 is not None else None # (B, Head, hidden=Ci, C_sigma)

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_scores3 = attention_scores3 / math.sqrt(self.KV_size) if emb3 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_scores4 = attention_scores4 / math.sqrt(self.KV_size) if emb4 is not None else None # (B, Head, hidden=Ci, C_sigma)

        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_probs3 = self.softmax(self.psi(attention_scores3)) if emb3 is not None else None # (B, Head, hidden=Ci, C_sigma)
        attention_probs4 = self.softmax(self.psi(attention_scores4)) if emb4 is not None else None # (B, Head, hidden=Ci, C_sigma)

        probs1 = attention_probs1
        probs2 = attention_probs2
        probs3 = attention_probs3
        probs4 = attention_probs4

        # print(attention_probs4.size())

        if self.vis:
            weights =  []
            weights.append(attention_probs1.mean(1)) # (B, hidden=Ci, C_sigma)
            weights.append(attention_probs2.mean(1)) # (B, hidden=Ci, C_sigma)
            weights.append(attention_probs3.mean(1)) # (B, hidden=Ci, C_sigma)
            weights.append(attention_probs4.mean(1)) # (B, hidden=Ci, C_sigma)
        else: 
            weights=None

        # weights ---> List of #(B, hidden=Ci, C_sigma) Tensors ---> len(weights) = Number Of Embeddings

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None
        attention_probs3 = self.attn_dropout(attention_probs3) if emb3 is not None else None
        attention_probs4 = self.attn_dropout(attention_probs4) if emb4 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2) # (B, Head, C_sigma, n_patch)
        # Cross Attentions
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None # (B, Head, hidden=Ci, n_patch)
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None # (B, Head, hidden=Ci, n_patch)
        context_layer3 = torch.matmul(attention_probs3, multi_head_V) if emb3 is not None else None # (B, Head, hidden=Ci, n_patch)
        context_layer4 = torch.matmul(attention_probs4, multi_head_V) if emb4 is not None else None # (B, Head, hidden=Ci, n_patch)

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None # (B, n_patch, hidden=Ci, Head)
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None # (B, n_patch, hidden=Ci, Head)
        context_layer3 = context_layer3.permute(0, 3, 2, 1).contiguous() if emb3 is not None else None # (B, n_patch, hidden=Ci, Head)
        context_layer4 = context_layer4.permute(0, 3, 2, 1).contiguous() if emb4 is not None else None # (B, n_patch, hidden=Ci, Head)

        context_layer1 = context_layer1.mean(dim=3) if emb1 is not None else None # (B, n_patch, hidden=Ci)
        context_layer2 = context_layer2.mean(dim=3) if emb2 is not None else None # (B, n_patch, hidden=Ci)
        context_layer3 = context_layer3.mean(dim=3) if emb3 is not None else None # (B, n_patch, hidden=Ci)
        context_layer4 = context_layer4.mean(dim=3) if emb4 is not None else None # (B, n_patch, hidden=Ci)

        O1 = self.out1(context_layer1) if emb1 is not None else None # (B, n_patch, hidden=Ci)
        O2 = self.out2(context_layer2) if emb2 is not None else None # (B, n_patch, hidden=Ci)
        O3 = self.out3(context_layer3) if emb3 is not None else None # (B, n_patch, hidden=Ci)
        O4 = self.out4(context_layer4) if emb4 is not None else None # (B, n_patch, hidden=Ci)
        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        O3 = self.proj_dropout(O3) if emb3 is not None else None
        O4 = self.proj_dropout(O4) if emb4 is not None else None
        if self.M_return:
            return O1,O2,O3,O4, weights, probs1, probs2, probs3, probs4
        else:
            return O1,O2,O3,O4, weights

class Mlp(nn.Module):
    def __init__(self,config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num,M_return=False):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.M_return = M_return
        self.attn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.attn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.attn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.attn_norm =  LayerNorm(config.KV_size,eps=1e-6)

        if self.M_return:
            self.channel_attn = Attention_org(config, vis, channel_num, M_return=self.M_return)
        else:
            self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.ffn_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.ffn_norm4 = LayerNorm(channel_num[3],eps=1e-6)
        self.ffn1 = Mlp(config,channel_num[0],channel_num[0]*expand_ratio)
        self.ffn2 = Mlp(config,channel_num[1],channel_num[1]*expand_ratio)
        self.ffn3 = Mlp(config,channel_num[2],channel_num[2]*expand_ratio)
        self.ffn4 = Mlp(config,channel_num[3],channel_num[3]*expand_ratio)


    def forward(self, emb1,emb2,emb3,emb4):
        embcat = []
        org1 = emb1 # (B, n_patches, hidden=Ci)
        org2 = emb2 # (B, n_patches, hidden=Ci)
        org3 = emb3 # (B, n_patches, hidden=Ci)
        org4 = emb4 # (B, n_patches, hidden=Ci)
        for i in range(4):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat,dim=2)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        cx3 = self.attn_norm3(emb3) if emb3 is not None else None
        cx4 = self.attn_norm4(emb4) if emb4 is not None else None
        emb_all = self.attn_norm(emb_all)

        if self.M_return:
            cx1,cx2,cx3,cx4, weights, probs1, probs2, probs3, probs4 = self.channel_attn(cx1,cx2,cx3,cx4,emb_all)
        else:
            cx1,cx2,cx3,cx4, weights = self.channel_attn(cx1,cx2,cx3,cx4,emb_all)

        cx1 = org1 + cx1 if emb1 is not None else None # (B, n_patch, hidden=Ci)
        cx2 = org2 + cx2 if emb2 is not None else None # (B, n_patch, hidden=Ci)
        cx3 = org3 + cx3 if emb3 is not None else None # (B, n_patch, hidden=Ci)
        cx4 = org4 + cx4 if emb4 is not None else None # (B, n_patch, hidden=Ci)

        org1 = cx1
        org2 = cx2
        org3 = cx3
        org4 = cx4
        
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x3 = self.ffn_norm3(cx3) if emb3 is not None else None
        x4 = self.ffn_norm4(cx4) if emb4 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None # (B, n_patch, hidden=Ci)
        x2 = self.ffn2(x2) if emb2 is not None else None # (B, n_patch, hidden=Ci)
        x3 = self.ffn3(x3) if emb3 is not None else None # (B, n_patch, hidden=Ci)
        x4 = self.ffn4(x4) if emb4 is not None else None # (B, n_patch, hidden=Ci)
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None
        x3 = x3 + org3 if emb3 is not None else None
        x4 = x4 + org4 if emb4 is not None else None

        if self.M_return:
            return x1, x2, x3, x4, weights, probs1, probs2, probs3, probs4
        else:
            return x1, x2, x3, x4, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num, M_return=True):
        super(Encoder, self).__init__()
                
        self.M_return = M_return
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0],eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1],eps=1e-6)
        self.encoder_norm3 = LayerNorm(channel_num[2],eps=1e-6)
        self.encoder_norm4 = LayerNorm(channel_num[3],eps=1e-6)

        for i in range(config.transformer["num_layers"]):
            if i==3 and self.M_return:
                layer = Block_ViT(config, vis, channel_num, M_return=self.M_return)
                self.layer.append(copy.deepcopy(layer))
            else:
                layer = Block_ViT(config, vis, channel_num)
                self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2,emb3,emb4):
        attn_weights = []
        for i,layer_block in enumerate(self.layer):
            if i==3 and self.M_return:
                emb1,emb2,emb3,emb4, weights, probs1, probs2, probs3, probs4 = layer_block(emb1,emb2,emb3,emb4)
                if self.vis:
                    attn_weights.append(weights)
            else:
                emb1,emb2,emb3,emb4, weights = layer_block(emb1,emb2,emb3,emb4)
                if self.vis:
                    attn_weights.append(weights)

        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        emb3 = self.encoder_norm3(emb3) if emb3 is not None else None
        emb4 = self.encoder_norm4(emb4) if emb4 is not None else None
        # weights ---> List of Lists of #(B, hidden=Ci, C_sigma) Tensors ---> len(weights) = Number Of Layers 
        if self.M_return:
            return emb1,emb2,emb3,emb4, attn_weights, probs1, probs2, probs3, probs4
        else:
            return emb1,emb2,emb3,emb4, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128, 256, 512], patchSize=[32, 16, 8, 4]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]
        self.patchSize_4 = patchSize[3]
        self.embeddings_1 = Channel_Embeddings(config,self.patchSize_1, img_size=img_size   , in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config,self.patchSize_2, img_size=img_size//2, in_channels=channel_num[1])
        self.embeddings_3 = Channel_Embeddings(config,self.patchSize_3, img_size=img_size//4, in_channels=channel_num[2])
        self.embeddings_4 = Channel_Embeddings(config,self.patchSize_4, img_size=img_size//8, in_channels=channel_num[3])

        self.M_return = False
        self.encoder = Encoder(config, vis, channel_num, M_return=self.M_return)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,scale_factor=(self.patchSize_1,self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,scale_factor=(self.patchSize_2,self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,scale_factor=(self.patchSize_3,self.patchSize_3))
        self.reconstruct_4 = Reconstruct(channel_num[3], channel_num[3], kernel_size=1,scale_factor=(self.patchSize_4,self.patchSize_4))

    def forward(self,en1,en2,en3,en4):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)
        emb4 = self.embeddings_4(en4)

        if self.M_return:
            encoded1, encoded2, encoded3, encoded4, attn_weights, probs1, probs2, probs3, probs4 = self.encoder(emb1,emb2,emb3,emb4)  # (B, n_patch, hidden)
        else:
            encoded1, encoded2, encoded3, encoded4, attn_weights = self.encoder(emb1,emb2,emb3,emb4)  # (B, n_patch, hidden)
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        x3 = self.reconstruct_3(encoded3) if en3 is not None else None
        x4 = self.reconstruct_4(encoded4) if en4 is not None else None

        x1 = x1 + en1  if en1 is not None else None
        x2 = x2 + en2  if en2 is not None else None
        x3 = x3 + en3  if en3 is not None else None
        x4 = x4 + en4  if en4 is not None else None

        if self.M_return:
            return x1, x2, x3, x4, attn_weights, probs1, probs2, probs3, probs4
        else:
            return x1, x2, x3, x4, attn_weights


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        x = torch.cat([skip_x, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UCTransNet_M(nn.Module):
    def __init__(self, config,n_channels=3, n_classes=1,img_size=224,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1), stride=(1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss
        self.pretrain = pretrain
    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x1,x2,x3,x4,att_weights, probs1, probs2, probs3, probs4 = self.mtc(x1,x2,x3,x4)

        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)


        logits = self.outc(x)


        if self.training:
            return logits, probs1, probs2, probs3, probs4
        else:
            return logits





