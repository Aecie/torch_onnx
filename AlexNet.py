import torch


class AlexNet(torch.nn.Module):
    def __init__(self, in_channels: int=3, out_features: int=1000):
        super().__init__()
        self.__fixed_out_size__ = 256
        self.__fixed_size__ = 6
        self.__feature_extractor__ = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0, dilation=1, ceil_mode=False),
            torch.nn.Conv2d(in_channels=192, out_channels=384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=384, out_channels=self.__fixed_out_size__, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(in_channels=self.__fixed_out_size__, out_channels=self.__fixed_out_size__, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=0, dilation=1, ceil_mode=False),
            torch.nn.AdaptiveAvgPool2d(output_size=(self.__fixed_size__, self.__fixed_size__))
        )

        self.__classifier__ = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(p=.5, inplace=False),
            torch.nn.Linear(in_features=self.__fixed_out_size__*(self.__fixed_size__**2), out_features=4096, bias=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=.5, inplace=False),
            torch.nn.Linear(in_features=4096, out_features=4096, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=4096, out_features=out_features, bias=True),
        )
    def forward(self, x):
        x1 = self.__feature_extractor__(x)
        y = self.__classifier__(x1)
        return y