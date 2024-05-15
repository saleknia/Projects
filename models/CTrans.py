from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
import numpy as np

class Spatial_Embeddings(nn.Module):
    """
    Construct the embeddings from patch, position embeddings.
    """
    def __init__(self,config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.transformer["embedding_channels"],
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.transformer["embedding_channels"]))

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
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

class Attention(nn.Module):
    def __init__(self, config,channel_num):
        super().__init__()
        self.KV_size = config.KV_size_S
        self.KV_size_C = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = self.KV_size // self.num_attention_heads

        self.query1 = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.query2 = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.query3 = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)

        self.key    = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.value  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.query_C  = nn.Linear(self.KV_size_C,  self.KV_size_C, bias=False)
        self.key_C    = nn.Linear(self.KV_size_C,  self.KV_size_C, bias=False)
        self.value_C  = nn.Linear(self.KV_size_C,  self.KV_size_C, bias=False)
        self.psi1 = nn.InstanceNorm2d(1)
        self.psi2  = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.attn_norm =  LayerNorm(config.KV_size_S,eps=1e-6)
        self.out1  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.out2  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)
        self.out3  = nn.Linear(config.transformer["embedding_channels"], config.transformer["embedding_channels"], bias=False)

        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb1,emb2,emb3, emb_C):
        #===============================================================================
        # CFA Module
        #===============================================================================
        Q_C = self.query_C(emb_C)
        K_C = self.key_C(emb_C)
        V_C = self.value_C(emb_C)

        attn = torch.matmul(Q_C.transpose(-1, -2), K_C)
        attn = attn.unsqueeze(1)
        ch_similarity_matrix = self.softmax(self.psi1(attn)).squeeze(1)
        ch_similarity_matrix = self.attn_dropout(ch_similarity_matrix)
        context_layer = torch.matmul(ch_similarity_matrix, V_C.transpose(-1, -2))
        T_hat = (context_layer.transpose(-1, -2))
        KV_S = torch.split(T_hat, self.KV_size_C//3, 2)
        KV_S = torch.cat(KV_S,dim=1)

        #===============================================================================
        # SSA Module
        #===============================================================================
        Q1 = self.query1(emb1)
        Q2 = self.query2(emb2)
        Q3 = self.query3(emb3)

        K = self.key(KV_S)
        V = self.value(KV_S)

        multi_head_Q1 = self.transpose_for_scores(Q1)
        multi_head_Q2 = self.transpose_for_scores(Q2)
        multi_head_Q3 = self.transpose_for_scores(Q3)

        multi_head_K = self.transpose_for_scores(K).transpose(-1, -2)
        multi_head_V = self.transpose_for_scores(V)

        attn1 = torch.matmul(multi_head_Q1, multi_head_K)
        attn2 = torch.matmul(multi_head_Q2, multi_head_K)
        attn3 = torch.matmul(multi_head_Q3, multi_head_K)

        sp_similarity_matrix1 = self.softmax(self.psi2(attn1))
        sp_similarity_matrix2 = self.softmax(self.psi2(attn2))
        sp_similarity_matrix3 = self.softmax(self.psi2(attn3))

        sp_similarity_matrix1 = self.attn_dropout(sp_similarity_matrix1)
        sp_similarity_matrix2 = self.attn_dropout(sp_similarity_matrix2)
        sp_similarity_matrix3 = self.attn_dropout(sp_similarity_matrix3)

        context_layer1 = torch.matmul(sp_similarity_matrix1, multi_head_V) 
        context_layer2 = torch.matmul(sp_similarity_matrix2, multi_head_V) 
        context_layer3 = torch.matmul(sp_similarity_matrix3, multi_head_V) 

        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer1.size()[:-2] + (self.KV_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape)
        context_layer2 = context_layer2.view(*new_context_layer_shape)
        context_layer3 = context_layer3.view(*new_context_layer_shape)

        O1 = self.out1(context_layer1)
        O2 = self.out2(context_layer2)
        O3 = self.out3(context_layer3)

        O1 = self.proj_dropout(O1)
        O2 = self.proj_dropout(O2)
        O3 = self.proj_dropout(O3)

        return O1,O2,O3

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
    def __init__(self, config, channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.attn_norm2 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.attn_norm3 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)

        self.attn_norm =  LayerNorm(config.KV_size_S,eps=1e-6)
        self.attn_norm_C=  LayerNorm(config.KV_size,eps=1e-6)
        self.channel_attn = Attention(config, channel_num)
        self.ffn_norm1 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.ffn_norm2 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.ffn_norm3 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)

        self.ffn1 = Mlp(config,config.transformer["embedding_channels"],config.transformer["embedding_channels"]*expand_ratio)
        self.ffn2 = Mlp(config,config.transformer["embedding_channels"],config.transformer["embedding_channels"]*expand_ratio)
        self.ffn3 = Mlp(config,config.transformer["embedding_channels"],config.transformer["embedding_channels"]*expand_ratio)

    def forward(self, emb1,emb2,emb3):
        embcat = []
        org1 = emb1
        org2 = emb2
        org3 = emb3

        for i in range(3):
            var_name = "emb"+str(i+1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_C = torch.cat(embcat,dim=2)

        cx1 = self.attn_norm1(emb1)
        cx2 = self.attn_norm2(emb2)
        cx3 = self.attn_norm3(emb3)

        emb_C = self.attn_norm_C(emb_C)

        cx1,cx2,cx3 = self.channel_attn(cx1,cx2,cx3, emb_C)

        cx1 = org1 + cx1
        cx2 = org2 + cx2
        cx3 = org3 + cx3

        org1 = cx1
        org2 = cx2
        org3 = cx3

        x1 = self.ffn_norm1(cx1)
        x2 = self.ffn_norm2(cx2)
        x3 = self.ffn_norm3(cx3)

        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)
        x3 = self.ffn3(x3)

        x1 = x1 + org1
        x2 = x2 + org2
        x3 = x3 + org3

        return x1, x2, x3

class Encoder(nn.Module):
    def __init__(self, config, channel_num):
        super(Encoder, self).__init__()

        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.encoder_norm2 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)
        self.encoder_norm3 = LayerNorm(config.transformer["embedding_channels"],eps=1e-6)

        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1,emb2,emb3):
        
        for layer_block in self.layer:
            emb1,emb2,emb3 = layer_block(emb1,emb2,emb3)
        
        emb1 = self.encoder_norm1(emb1)
        emb2 = self.encoder_norm2(emb2)
        emb3 = self.encoder_norm3(emb3)

        return emb1,emb2,emb3,

class DAT(nn.Module):
    def __init__(self, config, img_size, channel_num, patchSize):
        super().__init__()
        
        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.patchSize_3 = patchSize[2]

        self.embeddings_1 = Spatial_Embeddings(config,self.patchSize_1, img_size=img_size//4 , in_channels=channel_num[0])
        self.embeddings_2 = Spatial_Embeddings(config,self.patchSize_2, img_size=img_size//8 , in_channels=channel_num[1])
        self.embeddings_3 = Spatial_Embeddings(config,self.patchSize_3, img_size=img_size//16, in_channels=channel_num[2])

        self.encoder = Encoder(config, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,scale_factor=(self.patchSize_1,self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,scale_factor=(self.patchSize_2,self.patchSize_2))
        self.reconstruct_3 = Reconstruct(channel_num[2], channel_num[2], kernel_size=1,scale_factor=(self.patchSize_3,self.patchSize_3))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self,en1,en2,en3):

        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)
        emb3 = self.embeddings_3(en3)

        x1, x2, x3 = self.encoder(emb1,emb2,emb3)

        x1 = self.reconstruct_1(x1)
        x2 = self.reconstruct_2(x2)
        x3 = self.reconstruct_3(x3) 

        x1 = x1 + en1  
        x2 = x2 + en2  
        x3 = x3 + en3 

        return x1, x2, x3




