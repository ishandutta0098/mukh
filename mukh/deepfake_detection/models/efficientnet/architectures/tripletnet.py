"""
Video Face Manipulation Detection Through Ensemble of CNNs

Author Credits:

Image and Sound Processing Lab - Politecnico di Milano

Nicolò Bonettini
Edoardo Daniele Cannas
Sara Mandelli
Luca Bondi
Paolo Bestagini

GitHub: https://github.com/polimi-ispl/icpr2020dfdc
License: GNU GENERAL PUBLIC LICENSE
"""

from . import fornet
from .fornet import FeatureExtractor


class TripletNet(FeatureExtractor):
    """
    Template class for triplet net
    """

    def __init__(self, feat_ext: FeatureExtractor):
        super(TripletNet, self).__init__()
        self.feat_ext = feat_ext()
        if not hasattr(self.feat_ext, "features"):
            raise NotImplementedError(
                "The provided feature extractor needs to provide a features() method"
            )

    def features(self, x):
        return self.feat_ext.features(x)

    def forward(self, x1, x2, x3):
        x1 = self.features(x1)
        x2 = self.features(x2)
        x3 = self.features(x3)
        return x1, x2, x3


class EfficientNetB4(TripletNet):
    def __init__(self):
        super(EfficientNetB4, self).__init__(feat_ext=fornet.EfficientNetB4)


class EfficientNetAutoAttB4(TripletNet):
    def __init__(self):
        super(EfficientNetAutoAttB4, self).__init__(
            feat_ext=fornet.EfficientNetAutoAttB4
        )
