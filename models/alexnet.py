import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo


# __all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class Model(nn.Module):

    def __init__(self, num_classes=1000):
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

        self.fc1 = nn.Linear(num_classes + 2, 2)
        # self.fc1 = nn.Linear(num_classes, 2)


        # self.fc1 = nn.Linear(num_classes + 2, 502)
        # self.fc2 = nn.Linear(502, 2)

    # def forward(self, x, y):
    #     # x = x.float()
    #     # y = y.float()
    #
    #     x = self.features(x)
    #     x = x.view(x.size(0), 256 * 6 * 6)
    #     x = self.classifier(x)
    #     # x = torch.cat([x, y], dim=1)
    #
    #     x = self.relu(x)
    #     x = self.fc1(x)
    #     # x = self.fc2(x)
    #     return x

    def forward(self, x, y):
        # x = x.float()
        # y = y.float()

        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = torch.cat([x, y], dim=1)

        x = self.relu(x)
        x = self.fc1(x)
        # x = self.fc2(x)
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))
    return model

