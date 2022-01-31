import torch
from einops import rearrange
from torch import einsum
import torch.utils.model_zoo as model_zoo
import torch.nn as nn


def _conv2d1x1(in_channels, out_channels, stride=1):

    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                         nn.BatchNorm2d(out_channels))


def relative_to_absolute(q):
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


def rel_pos_emb_1d(q, rel_emb, shared_heads):
    if shared_heads:
        emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
    else:
        emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
    return relative_to_absolute(emb)


class RelPosEmb1D(nn.Module):
    def __init__(self, tokens, dim_head, heads=None):

        super().__init__()
        scale = dim_head ** -0.5
        self.shared_heads = heads if heads is not None else True
        if self.shared_heads:
            self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale)
        else:
            self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale)

    def forward(self, q):
        return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)


class RelPosEmb2D(nn.Module):
    def __init__(self, feat_map_size, dim_head, heads=None):
        super().__init__()
        self.h, self.w = feat_map_size  # height , width
        self.total_tokens = self.h * self.w
        self.shared_heads = heads if heads is not None else True

        self.emb_w = RelPosEmb1D(self.h, dim_head, heads)
        self.emb_h = RelPosEmb1D(self.w, dim_head, heads)

    def expand_emb(self, r, dim_size):
        # Decompose and unsqueeze dimension
        r = rearrange(r, 'b (h x) i j -> b h x () i j', x=dim_size)
        expand_index = [-1, -1, -1, dim_size, -1, -1]  # -1 indicates no expansion
        r = r.expand(expand_index)
        return rearrange(r, 'b h x1 x2 y1 y2 -> b h (x1 y1) (x2 y2)')

    def forward(self, q):
        assert self.total_tokens == q.shape[2], f'Tokens {q.shape[2]} of q must \
        be equal to the product of the feat map size {self.total_tokens} '

        # out: [batch head*w h h]
        r_h = self.emb_w(rearrange(q, 'b h (x y) d -> b (h x) y d', x=self.h, y=self.w))
        r_w = self.emb_h(rearrange(q, 'b h (x y) d -> b (h y) x d', x=self.h, y=self.w))
        q_r = self.expand_emb(r_h, self.h) + self.expand_emb(r_w, self.h)
        return q_r  # q_r transpose in figure 4 of the paper


class BottleneckAttention(nn.Module):
    def __init__(
            self,
            dim,
            fmap_size,
            heads=4,
            dim_head=None,
            content_positional_embedding=True
    ):

        super().__init__()
        self.heads = heads
        self.dim_head = (int(dim / heads)) if dim_head is None else dim_head
        self.scale = self.dim_head ** -0.5
        self.fmap_size = fmap_size
        self.content_positional_embedding = content_positional_embedding

        self.to_qkv = nn.Conv2d(dim, heads * self.dim_head * 3, 1, bias=False)

        self.height = self.fmap_size[0]
        self.width = self.fmap_size[1]

        if self.content_positional_embedding:
            self.pos_emb2D = RelPosEmb2D(feat_map_size=fmap_size, dim_head=self.dim_head)

    def forward(self, x):
        assert x.dim() == 4, f'Expected 4D tensor, got {x.dim()}D tensor'

        qkv = self.to_qkv(x)
        q, k, v = tuple(rearrange(qkv, 'b (d k h ) x y  -> k b h (x y) d', k=3, h=self.heads))
        dot_prod = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if self.content_positional_embedding:
            dot_prod = dot_prod + self.pos_emb2D(q)
        attention = torch.softmax(dot_prod, dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attention, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=self.height, y=self.width)
        return out


class BottleneckBlock(nn.Module):
    def __init__(self, *, in_channels,
                 fmap_size,
                 out_channels=2048,
                 proj_factor=4,
                 heads=4,
                 dim_head=None,
                 pooling=False,
                 content_positional_embedding=True):
        super().__init__()
        bottleneck_dimension = out_channels // proj_factor  # contraction_channels 512
        mhsa_out_channels = bottleneck_dimension if dim_head is None else dim_head * heads

        contraction = _conv2d1x1(in_channels, bottleneck_dimension)

        bot_mhsa = BottleneckAttention(
            dim=bottleneck_dimension,
            fmap_size=fmap_size,
            heads=heads,
            dim_head=dim_head,
            content_positional_embedding=content_positional_embedding)

        pool_layer = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2)) if pooling else nn.Identity()

        expansion = _conv2d1x1(mhsa_out_channels, out_channels)

        self.block = nn.Sequential(contraction,
                                   nn.ReLU(),
                                   bot_mhsa,
                                   pool_layer,
                                   nn.BatchNorm2d(mhsa_out_channels),
                                   nn.ReLU(),
                                   expansion)  # no relu after expansion

        # TODO find init_zero=True tf param for batch norm

        if pooling or in_channels != out_channels:
            stride = 2 if pooling else 1
            self.shortcut = nn.Sequential(
                _conv2d1x1(in_channels, out_channels, stride=stride),
                nn.ReLU())
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        print('forward..', x.shape)
        return self.block(x) + self.shortcut(x)


class BottleneckModule(nn.Module):
    def __init__(self, *, in_channels,
                 fmap_size,
                 out_channels=2048,
                 proj_factor=4,
                 heads=4,
                 dim_head=None,
                 pooling=True,
                 content_positional_embedding=True,
                 num_layers=3,  # default
                 ):
        super().__init__()
        block_list = []
        for i in range(num_layers):
            if i == 0:
                feat_map = fmap_size
                if pooling:
                    pool = True
            else:
                pool = False
                if pooling:
                    in_channels = out_channels
                    feat_map = (fmap_size[0] // 2, fmap_size[1] // 2)

            block_list.append(BottleneckBlock(in_channels=in_channels,
                                              fmap_size=feat_map,
                                              out_channels=out_channels,
                                              proj_factor=proj_factor,
                                              heads=heads,
                                              dim_head=dim_head,
                                              pooling=pool,
                                              content_positional_embedding=content_positional_embedding))
        self.model = nn.Sequential(*block_list)

    def forward(self, x):
        return self.model(x)


def conv3x3(in_ch, out_ch, stride=1, dilation=1, padding=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, dilation=dilation, padding=padding, bias=False)


def conv1x1(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_ch, out_ch, stride=1, dilation=1, is_botnet_layer=False, downsample=None):
        super(BottleBlock, self).__init__()
        self.conv1 = conv1x1(in_ch, out_ch)
        self.bn1 = nn.BatchNorm2d(out_ch, momentum=1, affine=True)
        if is_botnet_layer:
            self.conv2 = BottleneckBlock(in_channels=out_ch, fmap_size=(28, 28), heads=4, out_channels=out_ch, pooling=False)
        else:
            self.conv2 = conv3x3(out_ch, out_ch, stride=stride, padding=dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_ch, momentum=1, affine=True)
        self.conv3 = conv1x1(out_ch, out_ch * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_ch * self.expansion, momentum=1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsmaple = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsmaple is not None:
            residual = self.downsmaple(x)

        out = out + residual
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self, in_ch, block, layers, os=16, pretrained=False, level='net50'):
        super(Net, self).__init__()
        self.level = level
        self.inplanes = 64
        if os == 16:
            stride = [1, 2, 2, 1]
            dilation = [1, 1, 1, 2]
            dilation_ratio = [1, 2, 4]
        elif os == 8:
            stride = [1, 2, 1, 1]
            dilation = [1, 1, 1, 1]
            dilation_layer3 = [1, 2, 5, 1, 2, 5]
            dilation_layer4 = [1, 2, 5]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=1, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, layers[0], stride=stride[0], dilation=1, is_botnet_layer=False)
        self.layer2 = self.make_layer(block, 128, layers[1], stride=stride[1], dilation=1, is_botnet_layer=False)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=stride[2], dilation=1, is_botnet_layer=True)

        self.init_weight()


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        stem_feature = x
        x = self.maxpool(x)
        x = self.layer1(x)
        stage1_feature = x

        x1 = self.layer2(x)
        stage2_feature = x1
        x2 = self.layer3(x1)
        stage3_feature = x2
        x = torch.cat([x1, x2], dim=1)
        # x_support = self.layer_support(x)
        # x_query = self.layer_query(x)

        return stem_feature, stage1_feature, stage2_feature, stage3_feature, x

    def make_layer(self, block, in_ch, blocks, stride=1, dilation=1, is_botnet_layer=False):
        downsample = None
        if stride != 1 or self.inplanes != in_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, in_ch * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_ch * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, in_ch, stride, dilation, is_botnet_layer, downsample))
        self.inplanes = in_ch * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, in_ch, is_botnet_layer=is_botnet_layer))

        return nn.Sequential(*layers)

    def make_dilation_layer(self, block, in_ch, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != in_ch * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, in_ch * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(in_ch * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, in_ch, stride, dilation=blocks[0] * dilation, downsample=downsample))
        self.inplanes = in_ch * block.expansion
        # print(self.inplanes)
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, in_ch, stride=1, dilation=blocks[i] * dilation))

        return nn.Sequential(*layers)


    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def backbone(in_ch=3, os=8, pretrained=False):
    model = Net(in_ch=in_ch, block=BottleBlock, layers=[3, 4, 6, 3], os=os, pretrained=pretrained)

    return model



