import torch.nn as nn
import torch

class RotNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(RotNet, self).__init__()

        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()

        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.fc = nn.Linear(feature_size, num_classes)
        self.rot_fc = nn.Linear(feature_size, 4)

    def forward(self, x, return_rot_logits=False):
        _, feature = self.backbone(x, return_feature=True)

        logits = self.fc(feature)
        rot_logits = self.rot_fc(feature)

        if return_rot_logits:
            return logits, rot_logits
        else:
            return logits

# FedOV+RotPred
class FedOVRotNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super(FedOVRotNet, self).__init__()

        self.backbone = backbone
        if hasattr(self.backbone, 'fc'):
            # remove fc otherwise ddp will
            # report unused params
            self.backbone.fc = nn.Identity()

        try:
            feature_size = backbone.feature_size
        except AttributeError:
            feature_size = backbone.module.feature_size

        self.fc = nn.Linear(feature_size, num_classes+1)
        self.rot_fc = nn.Linear(feature_size, 4)
        # self.known_fc = nn.Linear(feature_size, 2)
        self.use_rotpred = True

    def forward(self, x, return_rot_logits=False):
        _, feature = self.backbone(x, return_feature=True)

        logits = self.fc(feature)
        rot_logits = self.rot_fc(feature)
        # known_logits = self.known_fc(feature)
        # max_logit = torch.max(logits,dim=1)[0]
        # known_logits = torch.cat((known_logits, max_logit.unsqueeze(1)),dim=1)
        # known_logits = known_logits / torch.norm(known_logits, dim=1).unsqueeze(1)

        if return_rot_logits:
            return logits, rot_logits #, known_logits
        else:
            return logits
