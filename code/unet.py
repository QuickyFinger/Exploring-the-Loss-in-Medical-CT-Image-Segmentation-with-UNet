import torch
import torch.nn as nn
import torch.nn.functional as F

def conv1x1(in_channels, out_channels, groups=1):
    '''
    a convolutional layer with 1x1 kernel size
    :param in_channels: (int) number of channels in the input images
    :param out_channels: (int) number of channels produced by the convolution
    :param groups: (int, optional) number of blocked collections from input channels to the output channels, default 1
    :return: a 1x1 convolution layer
    '''
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=1,
                     groups=groups,
                     stride=1)

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=True, groups=1):
    '''
    a convolutional layer with 3x3 kernel size
    :param in_channels: (int) number of channels in the input images.
    :param out_channels: (int) number of channels produced by the convolution
    :param stride: (int or tuple, optional) stride of the convolution, default 1
    :param padding: (int or tuple, optional) zero-padding added to both sides of the input, default 0
    :param bias: (bool, optional) if true, add learnable bias to the output, default true.
    :param groups: (int, optional) number of blocked collections from input channels to the output channels, default 1
    :return: a 3x3 convolution layer
    '''
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size=3,
                     stride=stride,
                     padding=padding,
                     bias=bias,
                     groups=groups)

def upconv2x2(in_channels, out_channels, mode='transpose'):
    '''
    up convolution
    :param in_channels: (int) number of channels in the input images
    :param out_channels: (int) number of channels produced by the convolution
    :param mode: tranpose convolution or up sampling
    :return: a 2x2 up convolution layer
    '''
    # output = (input - 1) * stride + outputpadding - 2 * padding + kernelsize
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels,
                                  out_channels,
                                  kernel_size=2,
                                  stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
                        conv1x1(in_channels, out_channels))

class DownConv(nn.Module):
    '''
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    '''
    def __init__(self, in_channels, out_channels, pooling=True):
        super(DownConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        before_pool = x

        if self.pooling:
            x = self.pool(x)

        return x, before_pool

class UpConv(nn.Module):
    '''
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    '''
    def __init__(self,
                 in_channels,
                 out_channels,
                 merge_mode='concat',
                 up_mode='transpose'):
        super(UpConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.up_mode = up_mode
        self.upconv = upconv2x2(self.in_channels,
                                self.out_channels,
                                mode=self.up_mode)
        if self.merge_mode == 'concat':
            self.conv1 = conv3x3(2*self.out_channels,
                                 self.out_channels)
        else:
            # number of input channels to conv2 is same
            self.conv1 = conv3x3(self.out_channels, self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, from_down, from_up):
        '''
        Forwar pass
        :param from_down: tensor from the encode pathway
        :param from_up: upconv'd tensor from the decoder pathway
        :return:
        '''
        from_up = self.upconv(from_up)
        if self.merge_mode == 'concat':
            x = torch.cat((from_up, from_down), 1)
        else:
            x = from_up + from_down
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class UNet(nn.Module):
    '''
    UNet class is based on https://arxiv.org/abs/1505.04597
    The U-Net is a convolutional encoder-decoder neural network.
    Contextual spatial information (from the encoding,
    expensive path way) about an input tensor is merged with
    information representing the localizaiton of details
    (from the encoding, compressive pathway).
    Modifications to the original paper:
    (1) padding is used in 3x3 convolutions to prevent loss of border pixels
    (2) merging outputs does not require cropping due to (1)
    (3) residual connections can be used by specifying
        UNet (merge_mode='add')
    (4) if non-parametric upsamping is used in the decoder
        pathway (specified by upmode='upsample'), then an additional 1x1 2d
        convolution occurs after upsampling to reduce channel dimensionality
        by a factor of 2. This channel halving happens with the convolution
        in the transpose convolution (specified by upmode='transpose')
    '''
    def __init__(self, num_classes, in_channels=1, depth=5,
                 start_filts=64, up_mode='transpose', merge_mode='concat'):
        '''
        :param num_classes: int, number of classes
        :param in_channels: int, number of channels in the input tensor.
                    default is 3 for RGB images.
        :param depth: int, number of MaxPools in the U-Net.
        :param start_filts: int, number of convolutional filters for the first conv.
        :param up_mode:  string, type of upconvolution. Choices: 'transpose' for
                    transpose convolution or 'upsample' for nearest neigbour
                    upsampling.
        :param merge_mode: string, type of merge. Choices: 'concat' for
                    concatenation, 'add' for addition.
        '''
        super(UNet, self).__init__()
        if up_mode in ('transpose', 'upsample'):
            self.up_mode = up_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for "
                             "upsampling. Only \"transpose\" and "
                             "\"upsample\" are allowed.".format(up_mode))
        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(up_mode))
        # NOTE: up_mode 'upsample' is incompatible with merge_mode 'add'
        if self.up_mode == 'upsample' and self.merge_mode == 'add':
            raise ValueError("up_mode \"upsample\" is incompatible "
                             "with merge_mode \"add\" at the moment "
                             "because it doesn't make sense to use "
                             "nearest neigbour to reduce "
                             "depth channels (by half). ")
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.start_filts = start_filts
        self.depth = depth
        self.down_convs = []
        self.up_convs = []

        # create the encoder pathway and add to a list
        for i in range(depth):
            ins = self.in_channels if i == 0 else outs
            outs = self.start_filts*(2**i)
            pooling = True if i < depth-1 else False
            down_conv = DownConv(ins, outs, pooling=pooling)
            self.down_convs.append(down_conv)

        # create the decoder pathway and add to a list
        # -careful! decoding only requires depth-1 blocks
        for i in range(depth-1):
            ins = outs
            outs = ins // 2
            up_conv = UpConv(ins, outs, up_mode=up_mode,
                             merge_mode=merge_mode)
            self.up_convs.append(up_conv)
        #self.conv_final = conv1x1(outs, self.num_classes)
        self.conv_final = conv1x1(outs, 1)
        # add the list of modules to current module
        self.down_convs = nn.ModuleList(self.down_convs)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            nn.init.constant(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        encoder_outs = []

        # encoder pathway, save outputs for merging
        for i, module in enumerate(self.down_convs):
            x, before_pool = module(x)
            encoder_outs.append(before_pool)

        for i, module in enumerate(self.up_convs):
            before_pool = encoder_outs[-(i+2)]
            x = module(before_pool, x)

        # No sofrmax is used. This means you need to use
        # nn.CrossEntropyLoss is you training script,
        # as this module includes a softmax already.
        x = self.conv_final(x)
        return x
