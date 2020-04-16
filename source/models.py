import torch
import torch.nn as nn


def cnn_layers(in_channels, batch_norm=False):
    # fmt: off
    config = [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M"
    ]
    # fmt: on

    layers = []

    for v in config:

        # maxpool
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        # conv2d layers
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

            if batch_norm:
                layers += [
                    conv2d,
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]

            # update in_channels
            in_channels = v

    return nn.Sequential(*layers)


def fc_layers(num_classes):
    # fully connected layers of vgg

    return nn.Sequential(
        nn.Linear(512 * 7 * 7, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(512, num_classes),
    )


class vgg16(nn.Module):
    def __init__(self, num_classes, channels=3):
        """
        vgg16 module

        parameters -------------------------
        - num_classes   -   number of outputs to predict
        - channels      -   number of input channels (eg. RGB:3)
        """

        # inheriting from module class
        super(vgg16, self).__init__()

        # metadata
        self.name = "vgg16"
        self.num_classes = num_classes

        # layers
        self.features = cnn_layers(channels)
        self.classifier = fc_layers(num_classes)

        self.init_weights()

        # transfer to gpu if cuda found
        if torch.cuda.is_available():
            self.cuda()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def memory_usage(self):
        """
        Get the total parameters of the model
        """

        def multiply_iter(iterable):
            res = 1
            for x in iterable:
                res *= x
            return res

        def add_params(parameter):
            res = 0
            for x in parameter:
                res += multiply_iter(x.shape)
            return res

        feat = add_params(self.features.parameters())
        clsf = add_params(self.classifier.parameters())
        total = feat + clsf

        mb_f = 4 / 1024 ** 2

        print("Conv   : {0}".format(feat))
        print("FC     : {0}".format(clsf))
        print("-----------------")
        print("Total  : {0}".format(total))
        print("Memory : {0:.2f}MB".format(total * mb_f))
        print("")

    def init_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def load_weights(self, saved_dict, ignore_keys=[]):

        # indexable ordered dict
        state_dict = self.state_dict()
        saved_dict = list(saved_dict.items())

        # update state_dict where pretrained dict is similar
        for i, (key, val) in enumerate(state_dict.items()):

            space = " " * (25 - len(str(key))) + " "
            n_val = saved_dict[i][1]

            if (
                key not in ignore_keys
                and val.shape == n_val.shape
            ):
                state_dict[key] = n_val
                print("   " + str(key) + space + "Loaded")

            else:
                print("   " + str(key) + space + "Ignored")

        self.load_state_dict(state_dict)

    def freeze_cnn_layers(self, except_last=0):

        num_params = len(list(self.features.parameters()))
        state_keys = [key for key in self.features.state_dict()]

        for i, param in enumerate(self.features.parameters()):

            key = state_keys[i]
            space = " " * (25 - len(str(key))) + " "

            if num_params - i > except_last:
                param.requires_grad = False
                print("   " + str(key) + space + "Frozen")

            else:
                param.requires_grad = True
                print("   " + str(key) + space + "Active")

