import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

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

    def __init__(self, in_channels, out_channels, activation='ReLU',kernel_size=3, padding=1, stride=1, dilation=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class BatchNorm(nn.Module):
    """([BN] => ReLU)"""

    def __init__(self, in_channels, activation='ReLU'):
        super(BatchNorm, self).__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.norm(x)
        return self.activation(out)

class Conv(nn.Module):
    """(convolution)"""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation)

    def forward(self, x):
        out = self.conv(x)
        return out

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        # self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        # out = self.maxpool(x)
        out = x
        return self.nConvs(out)




BATCH_NORM_DECAY = 1 - 0.9  # pytorch batch norm `momentum = 1 - counterpart` of tensorflow
BATCH_NORM_EPSILON = 1e-5


def get_act(activation):
    """Only supports ReLU and SiLU/Swish."""
    assert activation in ['relu', 'silu']
    if activation == 'relu':
        return nn.ReLU()
    else:
        return nn.Hardswish()  # TODO: pytorch's nn.Hardswish() v.s. tf.nn.swish


class BNReLU(nn.Module):
    """"""

    def __init__(self, out_channels, activation='relu', nonlinearity=True, init_zero=False):
        super(BNReLU, self).__init__()

        self.norm = nn.BatchNorm2d(out_channels, momentum=BATCH_NORM_DECAY, eps=BATCH_NORM_EPSILON)
        if nonlinearity:
            self.act = get_act(activation)
        else:
            self.act = None

        if init_zero:
            nn.init.constant_(self.norm.weight, 0)
        else:
            nn.init.constant_(self.norm.weight, 1)

    def forward(self, input):
        out = self.norm(input)
        if self.act is not None:
            out = self.act(out)
        return out


class RelPosSelfAttention(nn.Module):
    """Relative Position Self Attention"""

    def __init__(self, h, w, dim, relative=True, fold_heads=False):
        super(RelPosSelfAttention, self).__init__()
        self.relative = relative
        self.fold_heads = fold_heads
        self.rel_emb_w = nn.Parameter(torch.Tensor(2 * w - 1, dim))
        self.rel_emb_h = nn.Parameter(torch.Tensor(2 * h - 1, dim))

        nn.init.normal_(self.rel_emb_w, std=dim ** -0.5)
        nn.init.normal_(self.rel_emb_h, std=dim ** -0.5)

    def forward(self, q, k, v):
        """2D self-attention with rel-pos. Add option to fold heads."""
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        if self.relative:
            logits += self.relative_logits(q)
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def relative_logits(self, q):
        # Relative logits in width dimension.
        rel_logits_w = self.relative_logits_1d(q, self.rel_emb_w, transpose_mask=[0, 1, 2, 4, 3, 5])
        # Relative logits in height dimension
        rel_logits_h = self.relative_logits_1d(q.permute(0, 1, 3, 2, 4), self.rel_emb_h,
                                               transpose_mask=[0, 1, 4, 2, 5, 3])
        return rel_logits_h + rel_logits_w

    def relative_logits_1d(self, q, rel_k, transpose_mask):
        bs, heads, h, w, dim = q.shape
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
        rel_logits = torch.reshape(rel_logits, [-1, heads * h, w, 2 * w - 1])
        rel_logits = self.rel_to_abs(rel_logits)
        rel_logits = torch.reshape(rel_logits, [-1, heads, h, w, w])
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat(1, 1, 1, h, 1, 1)
        rel_logits = rel_logits.permute(*transpose_mask)
        return rel_logits

    def rel_to_abs(self, x):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, length, 2*length - 1]
        Output: [bs, heads, length, length]
        """
        bs, heads, length, _ = x.shape
        col_pad = torch.zeros((bs, heads, length, 1), dtype=x.dtype).cuda()
        x = torch.cat([x, col_pad], dim=3)
        flat_x = torch.reshape(x, [bs, heads, -1]).cuda()
        flat_pad = torch.zeros((bs, heads, length - 1), dtype=x.dtype).cuda()
        flat_x_padded = torch.cat([flat_x, flat_pad], dim=2)
        final_x = torch.reshape(
            flat_x_padded, [bs, heads, length + 1, 2 * length - 1])
        final_x = final_x[:, :, :length, length - 1:]
        return final_x


class AbsPosSelfAttention(nn.Module):

    def __init__(self, W, H, dkh, absolute=True, fold_heads=False):
        super(AbsPosSelfAttention, self).__init__()
        self.absolute = absolute
        self.fold_heads = fold_heads

        self.emb_w = nn.Parameter(torch.Tensor(W, dkh))
        self.emb_h = nn.Parameter(torch.Tensor(H, dkh))
        nn.init.normal_(self.emb_w, dkh ** -0.5)
        nn.init.normal_(self.emb_h, dkh ** -0.5)

    def forward(self, q, k, v):
        bs, heads, h, w, dim = q.shape
        q = q * (dim ** -0.5)  # scaled dot-product
        logits = torch.einsum('bnhwd,bnpqd->bnhwpq', q, k)
        abs_logits = self.absolute_logits(q)
        if self.absolute:
            logits += abs_logits
        weights = torch.reshape(logits, [-1, heads, h, w, h * w])
        weights = F.softmax(weights, dim=-1)
        weights = torch.reshape(weights, [-1, heads, h, w, h, w])
        attn_out = torch.einsum('bnhwpq,bnpqd->bhwnd', weights, v)
        if self.fold_heads:
            attn_out = torch.reshape(attn_out, [-1, h, w, heads * dim])
        return attn_out

    def absolute_logits(self, q):
        """Compute absolute position enc logits."""
        emb_h = self.emb_h[:, None, :]
        emb_w = self.emb_w[None, :, :]
        emb = emb_h + emb_w
        abs_logits = torch.einsum('bhxyd,pqd->bhxypq', q, emb)
        return abs_logits


class GroupPointWise(nn.Module):
    """"""

    def __init__(self, in_channels, heads=4, proj_factor=1, target_dimension=None):
        super(GroupPointWise, self).__init__()
        if target_dimension is not None:
            proj_channels = target_dimension // proj_factor
        else:
            proj_channels = in_channels // proj_factor
        self.w = nn.Parameter(torch.Tensor(in_channels, heads, proj_channels // heads))

        nn.init.normal_(self.w, std=0.01)

    def forward(self, input):
        # dim order:  pytorch BCHW v.s. TensorFlow BHWC
        input = input.permute(0, 2, 3, 1).float()
        """
        b: batch size
        h, w : imput height, width
        c: input channels
        n: num head
        p: proj_channel // heads
        """
        out = torch.einsum('bhwc,cnp->bnhwp', input, self.w)
        return out


class MHSA(nn.Module):


    def __init__(self, in_channels, heads, curr_h, curr_w, pos_enc_type='relative', use_pos=True):
        super(MHSA, self).__init__()
        self.q_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.k_proj = GroupPointWise(in_channels, heads, proj_factor=1)
        self.v_proj = GroupPointWise(in_channels, heads, proj_factor=1)

        assert pos_enc_type in ['relative', 'absolute']
        if pos_enc_type == 'relative':
            self.self_attention = RelPosSelfAttention(curr_h, curr_w, in_channels // heads, fold_heads=True)
        else:
            raise NotImplementedError

    def forward(self, input):
        q = self.q_proj(input)
        k = self.k_proj(input)
        v = self.v_proj(input)

        o = self.self_attention(q=q, k=k, v=v)
        return o


class BotBlock(nn.Module):

    def __init__(self, in_dimension, curr_h, curr_w, proj_factor=4, activation='relu', pos_enc_type='relative',
                 stride=1, target_dimension=None):
        super(BotBlock, self).__init__()
        if stride != 1 or in_dimension != target_dimension:
            if target_dimension != 64:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
                    BNReLU(target_dimension, activation=activation, nonlinearity=True),
                    nn.Conv2d(target_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
                    BNReLU(target_dimension, activation=activation, nonlinearity=True),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
                    BNReLU(target_dimension, activation=activation, nonlinearity=True),
                )
        else:
            # self.shortcut = None
            if target_dimension != 64:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
                    BNReLU(target_dimension, activation=activation, nonlinearity=True),
                    nn.Conv2d(target_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
                    BNReLU(target_dimension, activation=activation, nonlinearity=True),
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
                    BNReLU(target_dimension, activation=activation, nonlinearity=True),
                )
        bottleneck_dimension = target_dimension // proj_factor
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=3,padding=1, stride=1),
            BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
        )

        self.mhsa = MHSA(in_channels=bottleneck_dimension, heads=4, curr_h=curr_h, curr_w=curr_w,
                         pos_enc_type=pos_enc_type)
        conv2_list = []
        if stride != 1:
            assert stride == 2, stride
            conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))  # TODO: 'same' in tf.pooling
        conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
        self.conv2 = nn.Sequential(*conv2_list)

        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=3,padding=1, stride=1),
            BNReLU(target_dimension, nonlinearity=False, init_zero=True),
        )
        self.last_act = get_act(activation)

        self.gamma = nn.parameter.Parameter(torch.zeros(1))

        self.curr_h = curr_h
        self.curr_w = curr_w

    def forward(self, x):
        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x
        Q_h = self.curr_h
        Q_w = self.curr_w
        N, C, H, W = x.shape
        P_h, P_w = H // Q_h, W // Q_w

        x = x.reshape(N * P_h * P_w, C, Q_h, Q_w)

        out = self.conv1(x)
        out = self.mhsa(out)
        out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
        out = self.conv2(out)
        out = self.conv3(out)

        N1, C1, H1, W1 = out.shape
        out = out.reshape(N, C1, int(H), int(W))

        out = out * self.gamma + shortcut

        # out += shortcut
        out = self.last_act(out)

        return out

# class BotBlock(nn.Module):

#     def __init__(self, in_dimension, curr_h=8, curr_w=8, proj_factor=4, activation='relu', pos_enc_type='relative',
#                  stride=1, target_dimension=None):
#         super(BotBlock, self).__init__()
#         # if stride != 1 or in_dimension != target_dimension:
#             # self.shortcut = nn.Sequential(
#             #     nn.Conv2d(in_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
#             #     BNReLU(target_dimension, activation=activation, nonlinearity=True),
#             #     nn.Conv2d(target_dimension, target_dimension, kernel_size=3,padding=1, stride=stride),
#             #     BNReLU(target_dimension, activation=activation, nonlinearity=True)
#             # )
#         #     self.shortcut = DownBlock(in_dimension, target_dimension, nb_Conv=2)
#         # else:
#         #     self.shortcut = None
#         self.shortcut = DownBlock(in_dimension, target_dimension, nb_Conv=2)

#         bottleneck_dimension = target_dimension // 3

#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=3,padding=1, stride=1, dilation=1),
#         #     BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
#         # )

#         # self.conv2 = nn.Sequential(
#         #     nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=3,padding=2, stride=1, dilation=2),
#         #     BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
#         # )

#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=3,padding=3, stride=1, dilation=3),
#         #     BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
#         # )

#         # self.conv4 = nn.Sequential(
#         #     nn.Conv2d(in_dimension, bottleneck_dimension, kernel_size=3,padding=4, stride=1, dilation=4),
#         #     BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True)
#         # )

#         self.conv1 = Conv(in_dimension, bottleneck_dimension ,kernel_size=3, padding=1, stride=1, dilation=1)
#         self.conv2 = Conv(in_dimension, bottleneck_dimension ,kernel_size=3, padding=2, stride=1, dilation=2)
#         self.conv3 = Conv(in_dimension, bottleneck_dimension ,kernel_size=3, padding=3, stride=1, dilation=3)

#         self.conv4 = Conv(bottleneck_dimension*3, target_dimension, kernel_size=1, padding=0, stride=1)
#         self.conv5 = Conv(target_dimension*2 ,target_dimension , kernel_size=1, padding=0, stride=1)
#         self.BN_1 = BatchNorm(in_channels=target_dimension, activation='ReLU')
#         self.BN_2 = BatchNorm(in_channels=target_dimension, activation='ReLU')

#         # self.mhsa = MHSA(in_channels=target_dimension, heads=4, curr_h=curr_h, curr_w=curr_w,pos_enc_type=pos_enc_type)

#         # conv2_list = []
#         # if stride != 1:
#         #     assert stride == 2, stride
#         #     conv2_list.append(nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)))  # TODO: 'same' in tf.pooling
#         # conv2_list.append(BNReLU(bottleneck_dimension, activation=activation, nonlinearity=True))
#         # self.conv2 = nn.Sequential(*conv2_list)

#         # self.conv3 = nn.Sequential(
#         #     nn.Conv2d(bottleneck_dimension, target_dimension, kernel_size=3,padding=1, stride=1),
#         #     BNReLU(target_dimension, nonlinearity=False, init_zero=True),
#         # )

#         # self.last_act = get_act(activation)


#     def forward(self, x):
#         # if self.shortcut is not None:
#         #     shortcut = self.shortcut(x)
#         # else:
#         #     shortcut = x

#         shortcut = self.shortcut(x)

#         out = torch.cat((self.conv1(x),self.conv2(x),self.conv3(x)),dim=1)
#         out = self.conv4(out)
#         out = self.BN_1(out)

#         out = torch.cat((out,shortcut),dim=1)
#         out = self.conv5(out)
#         out = self.BN_2(out)
        
#         # Q_h = Q_w = 8
#         # N, C, H, W = out.shape
#         # P_h, P_w = H // Q_h, W // Q_w

#         # out = out.reshape(N * P_h * P_w, C, Q_h, Q_w)

#         # out = self.mhsa(out).permute(0, 3, 1, 2) + out
#         # out = out.permute(0, 3, 1, 2)  # back to pytorch dim order
#         # N1, C1, H1, W1 = out.shape
#         # out = out.reshape(N, C1, int(H), int(W))

#         # out = torch.cat((out,shortcut),dim=1)
#         # out = self.conv4(out)
#         # out = self.BN_2(out)

#         # out = out + shortcut
#         # out = self.last_act(out)

#         return out

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x



def _make_bot_layer(ch_in, ch_out, w=8, h=8):
    W = w
    H = h
    dim_in = ch_in
    dim_out = ch_out

    stage5 = []

    stage5.append(
        BotBlock(in_dimension=dim_in, curr_h=H, curr_w=W, stride=1 , target_dimension=dim_out)
    )

    return nn.Sequential(*stage5)


class GT_U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = _make_bot_layer(ch_in=img_ch,ch_out=64)
        self.Conv2 = _make_bot_layer(ch_in=64,ch_out=128)
        self.Conv3 = _make_bot_layer(ch_in=128,ch_out=256)
        self.Conv4 = _make_bot_layer(ch_in=256,ch_out=512)
        self.Conv5 = _make_bot_layer(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = _make_bot_layer(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = _make_bot_layer(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = _make_bot_layer(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = _make_bot_layer(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1
