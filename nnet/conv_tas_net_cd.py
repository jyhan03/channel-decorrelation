import torch as th
import torch.nn as nn

import torch.nn.functional as F
from memonger import SublinearSequential  

EPS = 1e-8

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles
	
def CDblock(ref, aux):
    """
    Calculate the Channel Decorrelation between reference channel and the auxiliary channel.
    """
    assert aux.shape == ref.shape
    assert aux.dim() == 3
    
    def l2norm(mat, keepdim=True):
        return th.norm(mat, dim=-1, keepdim=keepdim)
    
    ref_zm = ref - th.mean(ref, -1, keepdim=True)
    aux_zm = aux - th.mean(aux, -1, keepdim=True)
    cos_dis = th.sum(ref_zm*aux_zm, -1, keepdim=True) / (l2norm(ref_zm)*l2norm(aux_zm)+EPS) 
    ones_dis = th.ones_like(cos_dis) 
    cda = F.softmax(th.stack([ones_dis, cos_dis]), 0)[0]

    return cda * aux

class ChannelWiseLayerNorm(nn.LayerNorm):
    """
    Channel wise layer normalization
    """

    def __init__(self, *args, **kwargs):
        super(ChannelWiseLayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        # LN
        x = super().forward(x)
        # N x C x T => N x T x C
        x = th.transpose(x, 1, 2)
        return x


class GlobalChannelLayerNorm(nn.Module):
    """
    Global channel layer normalization
    """

    def __init__(self, dim, eps=1e-05, elementwise_affine=True):
        super(GlobalChannelLayerNorm, self).__init__()
        self.eps = eps
        self.normalized_dim = dim
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.beta = nn.Parameter(th.zeros(dim, 1))
            self.gamma = nn.Parameter(th.ones(dim, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """
        x: N x C x T
        """
        if x.dim() != 3:
            raise RuntimeError("{} accept 3D tensor as input".format(
                self.__name__))
        # N x 1 x 1
        mean = th.mean(x, (1, 2), keepdim=True)
        var = th.mean((x - mean)**2, (1, 2), keepdim=True)
        # N x T x C
        if self.elementwise_affine:
            x = self.gamma * (x - mean) / th.sqrt(var + self.eps) + self.beta
        else:
            x = (x - mean) / th.sqrt(var + self.eps)
        return x

    def extra_repr(self):
        return "{normalized_dim}, eps={eps}, " \
            "elementwise_affine={elementwise_affine}".format(**self.__dict__)


def build_norm(norm, dim):
    """
    Build normalize layer
    LN cost more memory than BN
    """
    if norm not in ["cLN", "gLN", "BN"]:
        raise RuntimeError("Unsupported normalize layer: {}".format(norm))
    if norm == "cLN":
        return ChannelWiseLayerNorm(dim, elementwise_affine=True)
    elif norm == "BN":
        return nn.BatchNorm1d(dim)
    else:
        return GlobalChannelLayerNorm(dim, elementwise_affine=True)


class Conv1D(nn.Conv1d):
    """
    1D conv in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class ConvTrans1D(nn.ConvTranspose1d):
    """
    1D conv transpose in ConvTasNet
    """

    def __init__(self, *args, **kwargs):
        super(ConvTrans1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        """
        x: N x L or N x C x L
        """
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else th.unsqueeze(x, 1))
        if squeeze:
            x = th.squeeze(x)
        return x


class Conv1DBlock(nn.Module):
    """
    1D convolutional block:
        Conv1x1 - PReLU - Norm - DConv - PReLU - Norm - SConv
    """

    def __init__(self,
                 in_channels=256,
                 conv_channels=512,
                 kernel_size=3,
                 dilation=1,
                 norm="cLN",
                 causal=False):
        super(Conv1DBlock, self).__init__()
        # 1x1 conv
        self.conv1x1 = Conv1D(in_channels, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.lnorm1 = build_norm(norm, conv_channels)
        dconv_pad = (dilation * (kernel_size - 1)) // 2 if not causal else (
            dilation * (kernel_size - 1))
        # depthwise conv
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True)
        self.prelu2 = nn.PReLU()
        self.lnorm2 = build_norm(norm, conv_channels)
        # 1x1 conv cross channel
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        # different padding way
        self.causal = causal
        self.dconv_pad = dconv_pad

    def forward(self, x):
        y = self.conv1x1(x)
        y = self.lnorm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, :-self.dconv_pad]
        y = self.lnorm2(self.prelu2(y))
        y = self.sconv(y)
        x = x + y
        return x


class ConvTasNet(nn.Module):
    def __init__(self,
                 L=20,
                 N=256,
                 X=8,
                 R=4,
                 B=256,
                 H=512,
                 P=3,
                 norm="cLN",
                 non_linear="relu",
                 causal=False):
        super(ConvTasNet, self).__init__()
        self.L = L
        supported_nonlinear = {
            "relu": F.relu,
            "sigmoid": th.sigmoid,
            "softmax": F.softmax
        }
        if non_linear not in supported_nonlinear:
            raise RuntimeError("Unsupported non-linear function: {}",
                               format(non_linear))
        self.non_linear_type = non_linear
        self.non_linear = supported_nonlinear[non_linear]
        # n x S => n x N x T, S = 4s*8000 = 32000
        self.encoder_1d1 = Conv1D(1, N, L, stride=L // 2, padding=0) 
        self.encoder_1d2 = Conv1D(1, N, L, stride=L // 2, padding=0) 

        # before repeat blocks, always cLN
        self.ln = ChannelWiseLayerNorm(N)
        # n x N x T => n x B x T
        self.proj = Conv1D(N, B, 1)
        self.conv1d_block = Conv1DBlock(in_channels=B, conv_channels=H)  
        
        self.encoder_aux = Conv1D(1, N, L, stride=L // 2, padding=0) 
        self.conv1d_block_aux = Conv1DBlock(in_channels=N, conv_channels=H)  

        self.pred_linear = nn.Linear(N,101) # 101
        
        self.conv1d_block_layer1 = self._build_blocks_layer1(num_blocks=X,
                                                             in_channels=B,
                                                             conv_channels=H,
                                                             kernel_size=P,
                                                             norm=norm,
                                                             causal=causal)         
        
        # repeat blocks
        # n x B x T => n x B x T
        self.repeats = self._build_repeats(
            R,
            X,
            in_channels=B,
            conv_channels=H,
            kernel_size=P,
            norm=norm,
            causal=causal)

        self.mask = Conv1D(B, N, 1)
        # using ConvTrans1D: n x N x T => n x 1 x To
        # To = (T - 1) * L // 2 + L
        self.decoder_1d = ConvTrans1D(
            N, 1, kernel_size=L, stride=L // 2, bias=True)
        
    def _build_blocks_layer1(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(1, num_blocks)
        ]
        return SublinearSequential(*blocks)
    
    def _build_blocks(self, num_blocks, **block_kwargs):
        """
        Build Conv1D block
        """
        blocks = [
            Conv1DBlock(**block_kwargs, dilation=(2**b))
            for b in range(num_blocks)
        ]
        return SublinearSequential(*blocks)

    def _build_repeats(self, num_repeats, num_blocks, **block_kwargs):
        """
        Build Conv1D block repeats
        """
        repeats = [
            self._build_blocks(num_blocks, **block_kwargs)
            for r in range(1, num_repeats)
        ]
        return SublinearSequential(*repeats)

    def forward(self, mix1, mix2, aux, aux_len):
        if mix1.dim() == 1:
            mix1 = th.unsqueeze(mix1, 0)
            mix2 = th.unsqueeze(mix2, 0)
            aux = th.unsqueeze(aux, 0)
            aux_len = th.unsqueeze(aux_len, 0)

        w1 = F.relu(self.encoder_1d1(mix1))
        w2 = F.relu(self.encoder_1d2(mix2))
        w_cd = CDblock(w1, w2)

        aux = F.relu(self.encoder_aux(aux)) 
        aux = self.conv1d_block_aux(aux)
        aux_T = (aux_len - self.L) // (self.L // 2) + 1
        aux = th.sum(aux, -1)/aux_T.view(-1,1).float() 
        spk_pred = self.pred_linear(aux)

        w_cd = w_cd * th.unsqueeze(aux, -1)
        w = w_cd + w1 
        # n x B x T 
        y = self.proj(self.ln(w)) 
        y = self.conv1d_block(y) 
        
        y = y * th.unsqueeze(aux, -1) 
        # conv1d block layer 1
        y = self.conv1d_block_layer1(y)

        # n x B x T
        y = self.repeats(y) 
        e = self.mask(y)
        m = self.non_linear(e) 
        s = w * m
        return self.decoder_1d(s, squeeze=True), spk_pred


def foo_conv1d_block():
    nnet = Conv1DBlock(256, 512, 3, 20)
    print(param(nnet))


def foo_layernorm():
    C, T = 256, 20
    nnet1 = nn.LayerNorm([C, T], elementwise_affine=True)
    print(param(nnet1, Mb=False))
    nnet2 = nn.LayerNorm([C, T], elementwise_affine=False)
    print(param(nnet2, Mb=False))


def foo_conv_tas_net():
    mix1 = th.rand(4, 32000) 
    mix2 = th.rand(4, 32000) 
    aux = th.rand(4, 32000) 
    aux_len = th.tensor([31999, 20000, 32000, 32000])
    nnet = ConvTasNet(norm="cLN", causal=False)
    # print(nnet)
    print("ConvTasNet #param: {:.2f}".format(param(nnet)))
    s, _ = nnet(mix1, mix2, aux, aux_len)

    return s


if __name__ == "__main__":
    s = foo_conv_tas_net()
    # foo_conv1d_block()
    # foo_layernorm()
