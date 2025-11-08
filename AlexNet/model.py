import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2)
        self.maxpooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.5)

        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=96, kernel_size=(11, 11), stride=4, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=96, out_channels=256, kernel_size=(5, 5), groups=2, padding=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=384, kernel_size=(3, 3), groups=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=384, out_channels=384, kernel_size=(3, 3), groups=2, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=384, out_channels=256, kernel_size=(3, 3), groups=2, padding=1
        )

        self.linear1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.linear2 = nn.Linear(in_features=4096, out_features=4096)
        self.linear3 = nn.Linear(in_features=4096, out_features=1000)

        self.network = nn.Sequential(
            self.conv1,
            self.activation,
            self.lrn,
            self.maxpooling,
            self.conv2,
            self.activation,
            self.lrn,
            self.maxpooling,
            self.conv3,
            self.activation,
            self.conv4,
            self.activation,
            self.conv5,
            self.activation,
            self.maxpooling,
            self.flatten,
            self.linear1,
            self.activation,
            self.dropout,
            self.linear2,
            self.activation,
            self.dropout,
            self.linear3,
        )

        self.init_weights()

    def init_weights(self):
        for name, parameters in self.network.named_parameters():
            # all weights have normal(0, 0.01)
            if name.endswith("weight"):
                parameters.data.normal_(mean=0, std=0.01)

            if name.endswith("bias"):
                # these are the biases for second, fourth, fifth conv layers and the linear layers
                if name.split(".")[0] in [4, 10, 12, 16, 19, 22]:
                    parameters.data.fill_(value=1)
                # all other biases
                else:
                    parameters.data.fill_(value=0)

    def forward(self, x):
        return self.network(x)
