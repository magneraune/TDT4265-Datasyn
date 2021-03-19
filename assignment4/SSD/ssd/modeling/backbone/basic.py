import torch
from torch import nn

def init_feature_map(num_filters, in_channels, out_channels, is_last_layer=False):
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
    return output
        

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
        
        self.out0 = init_feature_map(
            num_filters=num_filters*4, in_channels = output_channels[0],out_channels = output_channels[1]
        )
        self.out1 = init_feature_map(
            num_filters=num_filters*8, in_channels = output_channels[1],out_channels = output_channels[2]
        )
        self.out2 = init_feature_map(
            num_filters=num_filters*4, in_channels = output_channels[2],out_channels = output_channels[3]
        )
        self.out3 = init_feature_map(
            num_filters=num_filters*4, in_channels = output_channels[3],out_channels = output_channels[4]
        )
        self.out4 = init_feature_map(
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
        out_features.append(self.out0(out_features[0]))
        out_features.append(self.out1(out_features[1]))
        out_features.append(self.out2(out_features[2]))
        out_features.append(self.out3(out_features[3]))
        out_features.append(self.out4(out_features[4]))



        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
    
    
####### Improved BasicModel #########

def init_feature_map_improved(num_filters, in_channels, out_channels, is_last_layer=False):
    if is_last_layer:
        stride2=1
        padding2=0
    else:
        stride2=2
        padding2=1

    output = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.BatchNorm2d(num_filters),
        nn.ReLU(),
        nn.Conv2d(
            in_channels=num_filters,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride2,
            padding=padding2
        ),
    )
    return output

class BasicModel_improved(torch.nn.Module):
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
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_filters*2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_filters*4),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=num_filters*4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_filters*4),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*4,
                out_channels=output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1
            ),
        )
        
        self.out0 = init_feature_map_improved(
            num_filters=num_filters*4, in_channels = output_channels[0],out_channels = output_channels[1]
        )
        self.out1 = init_feature_map_improved(
            num_filters=num_filters*8, in_channels = output_channels[1],out_channels = output_channels[2]
        )
        self.out2 = init_feature_map_improved(
            num_filters=num_filters*4, in_channels = output_channels[2],out_channels = output_channels[3]
        )
        self.out3 = init_feature_map_improved(
            num_filters=num_filters*4, in_channels = output_channels[3],out_channels = output_channels[4]
        )
        self.out4 = init_feature_map_improved(
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
        out_features.append(self.out0(out_features[0]))
        out_features.append(self.out1(out_features[1]))
        out_features.append(self.out2(out_features[2]))
        out_features.append(self.out3(out_features[3]))
        out_features.append(self.out4(out_features[4]))



        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)
    
####### Maximum Improved BasicModel #########

class BasicModel_max_improved(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 75, 75),
    shape(-1, output_channels[1], 38, 38),
    shape(-1, output_channels[2], 19, 19),
    shape(-1, output_channels[3], 10, 10),
    shape(-1, output_channels[4], 5, 5),
    shape(-1, output_channels[5], 3, 3),
    shape(-1, output_channels[6], 1, 1)]
     where "output_channels" is the same as cfg.BACKBONE.OUT_CHANNELS
    """
    def __init__(self, cfg):
        super().__init__()
        output_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.output_channels = output_channels
        image_channels = cfg.MODEL.BACKBONE.INPUT_CHANNELS
        self.output_feature_shape = cfg.MODEL.PRIORS.FEATURE_MAPS
        print("image_channels: ", image_channels)

        num_filters = 32
        # self.feature_extractor now outputs 75x75 resolution
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_filters),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_filters*2),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=num_filters*2,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(num_filters*2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters*2,
                out_channels=output_channels[0],
                kernel_size=3,
                stride=1,
                padding=1
            ),
        )
        
        self.out0 = init_feature_map_improved(
            num_filters=num_filters*4, in_channels = output_channels[0],out_channels = output_channels[1]
        )
        self.out1 = init_feature_map_improved(
            num_filters=num_filters*8, in_channels = output_channels[1],out_channels = output_channels[2]
        )
        self.out2 = init_feature_map_improved(
            num_filters=num_filters*16, in_channels = output_channels[2],out_channels = output_channels[3]
        )
        self.out3 = init_feature_map_improved(
            num_filters=num_filters*4, in_channels = output_channels[3],out_channels = output_channels[4]
        )
        self.out4 = init_feature_map_improved(
            num_filters=num_filters*4, in_channels = output_channels[4],out_channels = output_channels[5]
        )
        self.out5 = init_feature_map_improved(
            num_filters=num_filters*4, in_channels = output_channels[5],out_channels = output_channels[6], is_last_layer=True
        )


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 75, 75),
            shape(-1, output_channels[1], 38, 38),
            shape(-1, output_channels[2], 19, 19),
            shape(-1, output_channels[3], 10, 10),
            shape(-1, output_channels[4], 5, 5),
            shape(-1, output_channels[5], 3, 3),
            shape(-1, output_channels[6], 1, 1)]
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
        out_features.append(self.layer5(out_features[5]))



        for idx, feature in enumerate(out_features):
            w, h = self.output_feature_shape[idx]
            expected_shape = (self.output_channels[idx], h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        return tuple(out_features)