import torch
from torch import nn

def init_layer(num_filters, in_channels, out_channels, is_last_layer=False):
        if is_last_layer:
            stride2=1
            padding2=0
        else:
            stride2=2
            padding2=1

        output = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=out_channels,
                kernel_size=3,
                stride=stride2,
                padding=padding2
            )
        )

class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS

        num_filters = 32 
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        
        self.layer0 = init_layer(
            num_filters=num_filters*4, in_channels = output_channels[0],out_channels = output_channels[1]
        )
        self.layer1 = init_layer(
            num_filters=num_filters*8, in_channels = output_channels[1],out_channels = output_channels[2]
        )
        self.layer2 = init_layer(
            num_filters=num_filters*4, in_channels = output_channels[2],out_channels = output_channels[3]
        )
        self.layer3 = init_layer(
            num_filters=num_filters*4, in_channels = output_channels[3],out_channels = output_channels[4]
        )
        self.layer4 = init_layer(
            num_filters=num_filters*4, in_channels = output_channels[4],out_channels = output_channels[5], is_last_layer=True
        )


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        out_features.append(self.feature_extractor(x))
        out_features.append(self.layer0(out_features[0]))
        out_features.append(self.layer1(out_features[1]))
        out_features.append(self.layer2(out_features[2]))
        out_features.append(self.layer3(out_features[3]))
        out_features.append(self.layer4(out_features[4]))



        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)

