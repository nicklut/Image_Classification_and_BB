'''
Skeleton model class. You will have to implement the classification and regression layers, along with the forward definition.
'''


from torch import nn
from torchvision import models

CLASS_NUM = 4 

class RCNN(nn.Module):
    def __init__(self):
        super(RCNN, self).__init__()

        # Pretrained backbone. If you are on the cci machine then this will not be able to automatically download
        #  the pretrained weights. You will have to download them locally then copy them over.
        #  During the local download it should tell you where torch is downloading the weights to, then copy them to 
        #  ~/.cache/torch/checkpoints/ on the supercomputer.
        resnet = models.resnet18(pretrained=True)

        # Remove the last fc layer of the pretrained network.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze backbone weights. 
        for param in self.backbone.parameters():
            param.requires_grad = False

        # TODO: Implement the fully connected layers for classification and regression.
        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, CLASS_NUM+1),
        )

        self.regression = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 4*CLASS_NUM),
        )

    def forward(self, x):
        # TODO: Implement forward. Should return a (batch_size x num_classes) tensor for classification
        #           and a (batch_size x num_classes x 4) tensor for the bounding box regression. 
        forward_prop = self.backbone(x)
        classification = self.classification(forward_prop)
        regression = self.regression(forward_prop)
        return classification, regression 



    """
    Add a classification layer that is fully-connected to the last remaining layer of the backbone
    network. If there are C classes, this classification layer should have C + 1 neurons, where
    class index 0 corresponds to the label nothing.
    """
