import torch
import numpy as np


class AdaptiveAvgPool2d_FittedONNX(torch.nn.Module):
    def __init__(self, output_size: tuple):
        super(AdaptiveAvgPool2d_FittedONNX, self).__init__()
        self.__output_size__ = np.array(output_size)
        self.__adaptive_padding__ = torch.nn.ParameterDict({
            'pad_height': torch.zeros(output_size),
            'pad_width': torch.zeros(output_size)
        })
        self.__avg_pooling__ = torch.nn.AvgPool2d((3, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
            output_size: shape (output_H, output_W)
            x: shape (batch size, channel, height, width)
        Returns:
            x: shape (batch size, channel, output_H, output_W)
        '''
        batch_size, channels, height, width = x.shape
        if (height < self.__output_size__[-2]):
            # height padding
            self.__adaptive_padding__['pad_height'] = torch.zeros((batch_size, channels, self.__output_size__[-2] - height, width))
            x = torch.cat((x, self.__adaptive_padding__['pad_height']), axis=-2)
            height = self.__output_size__[-2]
        if (width < self.__output_size__[-1]):
            # width padding
            self.__adaptive_padding__['pad_width'] = torch.zeros((batch_size, channels, height, self.__output_size__[-1] - width))
            x = torch.cat((x, self.__adaptive_padding__['pad_width']), axis=-1)
            width = self.__output_size__[-1]

        stride_size = np.floor(np.true_divide(x.shape[-2:], self.__output_size__)).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.__output_size__ - 1) * stride_size
        self.__avg_pooling__ = torch.nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = self.__avg_pooling__(x)
        return x


# m = AdaptiveAvgPool2d_FittedONNX(output_size=(28, 33))
# x = torch.randn(64, 3, 128, 128)

# m = m.cuda()
# x = x.cuda()

# pred_y = m(x)
# print(pred_y.shape)
# print(m)