import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


model_urls = {
	'net18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
	'net34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
	'net50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
	'net101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
	'net152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_ch, out_ch, stride=1, dilation=1, padding=1):
	return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, dilation=dilation, padding=padding, bias=False)


def conv1x1(in_ch, out_ch, stride=1):
	return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False)


class BottleBlock(nn.Module):
	expansion = 4

	def __init__(self, in_ch, out_ch, stride=1, dilation=1, downsample=None):
		super(BottleBlock, self).__init__()
		self.conv1 = conv1x1(in_ch, out_ch)
		self.bn1 = nn.BatchNorm2d(out_ch, momentum=1, affine=True)
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

		self.layer1 = self.make_layer(block, 64, layers[0], stride=stride[0], dilation=dilation[0])
		self.layer2 = self.make_layer(block, 128, layers[1], stride=stride[1], dilation=dilation[1])
		# self.layer3 = self.make_layer(block, 256, layers[2], stride=stride[2], dilation=dilation[2])
		self.layer3 = self.make_dilation_layer(block, 256, blocks=dilation_layer3, stride=stride[3], dilation=1)
		# # self.layer_query = self.make_dilation_layer(block, 512, dilation_ratio, stride=stride[3], dilation=dilation[3])

		self.init_weight()

		if pretrained:
			self.load_pretrained_model()
		
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

	def make_layer(self, block, in_ch, blocks, stride=1, dilation=1):
		downsample = None
		if stride != 1 or self.inplanes != in_ch * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, in_ch * block.expansion, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(in_ch * block.expansion)
			)

		layers = []
		layers.append(block(self.inplanes, in_ch, stride, dilation, downsample))
		self.inplanes = in_ch * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, in_ch))

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

	def load_pretrained_model(self):
		pretrained_dict = model_zoo.load_url(model_urls[self.level])
		model_dict = {}
		state_dict = self.state_dict()
		for k, v in pretrained_dict.items():
			if k in state_dict:
				model_dict[k] = v
		state_dict.update(model_dict)
		self.load_state_dict(state_dict)

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

