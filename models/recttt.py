import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import hflip, invert

class ReCTTT(nn.Module):
    def __init__(
            self,
            encoder,
            encoder_freeze,
            bottleneck,
            decoder,
            classifier,
    ) -> None:
        super(ReCTTT, self).__init__()
        self.encoder = encoder
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder
        self.classifier = classifier

    def forward(self, x):
        en, to_cls = self.encoder(x)
        with torch.no_grad():
            en_freeze, _ = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        ca = self.classifier(to_cls)
        return en_freeze + en, de, ca

    def train(self, mode=True):
        self.training = mode
        if mode is True:
            self.encoder.train(True)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
            self.classifier.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
            self.classifier.train(False)
        return self
        
    def test_train(self, layers_train=[True,True,True,True]):       
        self.encoder.layer1.train(layers_train[0])
        self.encoder.layer2.train(layers_train[1])
        self.encoder.layer3.train(layers_train[2])
        self.encoder.layer4.train(layers_train[3])
        self.encoder_freeze.train(False)
        self.bottleneck.train(False)
        self.decoder.train(False)
        self.classifier.train(True)

class ReCTTT2(nn.Module):
    def __init__(
            self,
            encoder,
            encoder2,
            encoder_freeze,
            bottleneck,
            decoder,
            classifier,
            classifier2
    ) -> None:
        super(ReCTTT2, self).__init__()
        #Encoder 1
        self.encoder = encoder
        self.encoder.fc = None
        #Encoder 2
        self.encoder2 = encoder2
        self.encoder2.fc = None

        #Teacher
        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.fc = None

        #Decoder
        self.bottleneck = bottleneck
        self.decoder = decoder

        #Classification
        self.classifier = classifier
        self.classifier2 = classifier2
        
    def forward(self, x):
        en, to_cls = self.encoder(x)
        en2, to_cls2 = self.encoder2(hflip(x))
        with torch.no_grad():
            en_freeze, _ = self.encoder_freeze(x)
            en_freeze_t, _ = self.encoder_freeze(hflip(x))
        en_2 = [torch.cat([a, b, c, d], dim=0) for a, b, c, d in zip(en, en_freeze, en_freeze_t, en2)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=4) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1], de[6][2], de[7][2], de[8][2], de[9][3], de[10][3], de[11][3]] #De1, Def, def(t), De2
        ca = self.classifier(to_cls)
        ca2 = self.classifier2(to_cls2)
        return en_freeze + en + en2 + en_freeze_t, de, ca, ca2

    def get_features(self, x):
        _, to_cls = self.encoder(x)
        return to_cls
    
    def train(self, mode=True):
        self.training = mode
        if mode is True:
            self.encoder.train(True)
            self.encoder2.train(True)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
            self.classifier.train(True)
            self.classifier2.train(True)
        else:
            self.encoder.train(False)
            self.encoder2.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
            self.classifier.train(False)
            self.classifier2.train(False)
        return self
    
    def test_train(self, layers_train=[True,True,True,True]):
        if layers_train[0] == layers_train[1] == layers_train[2] == layers_train[3]:
            self.encoder.train(True)
            self.encoder2.train(True)
        else:
            self.encoder.layer1.train(layers_train[0])
            self.encoder.layer2.train(layers_train[1])
            self.encoder.layer3.train(layers_train[2])
            self.encoder.layer4.train(layers_train[3])
            self.encoder2.layer1.train(layers_train[0])
            self.encoder2.layer2.train(layers_train[1])
            self.encoder2.layer3.train(layers_train[2])
            self.encoder2.layer4.train(layers_train[3])
        self.encoder_freeze.train(False)
        self.bottleneck.train(False)
        self.decoder.train(False)
        self.classifier.train(True)
        self.classifier2.train(True)

class ResnetCls(nn.Module):
    def __init__(
            self,
            encoder,
            classifier,
    ) -> None:
        super(ResnetCls, self).__init__()
        self.encoder = encoder
        #self.encoder.fc = None
        self.classifier = classifier

    def forward(self, x):
        _, to_cls = self.encoder(x)
        ca = self.classifier(to_cls)
        return ca

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            self.encoder.train(True)
            self.classifier.train(True)
        else:
            self.encoder.train(False)
            self.classifier.train(False)
        return self



'''
------------ LOSSES ----------
'''

class _Loss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, *input):
        raise NotImplementedError
    
class _WeightedLoss(_Loss):
    def __init__(self, weight=None, reduction='mean'):
        super(_WeightedLoss, self).__init__(reduction)
        self.register_buffer('weight', weight)

    def forward(self, *input):
        raise NotImplementedError

class DynamicMutualLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, gamma1, gamma2, margin=0, log=False):
        targets = targets.softmax(dim=1)
        real_targets = targets.argmax(1).clone().detach()
        total_loss = self.criterion_ce(inputs, real_targets)  # Pixel-level weight is always 1
        true_loss = total_loss.mean().item()

        outputs = inputs.softmax(dim=1).clone().detach()
        decision_current = outputs.argmax(1)  # N
        decision_pseudo = real_targets.clone().detach()  # N
        confidence_current = outputs.max(1).values  # N
        confidence_pseudo = targets.max(1).values.clone().detach()  # N added softmax otherwise pred >

        temp = decision_pseudo.unsqueeze(1).clone().detach()
        probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)

        dynamic_weights = torch.ones_like(decision_current).float()

        # Prepare indices
        disagreement = decision_current != decision_pseudo
        current_win = confidence_current > (confidence_pseudo + margin)

        # Agree
        indices = ~disagreement
        dynamic_weights[indices] = probabilities_current[indices] ** gamma1
        #if log == 0:
        #    print('total equal: {}'.format(sum(indices)))

        # Disagree (current model wins, do not learn!)
        indices = disagreement * current_win
        dynamic_weights[indices] = 0
        #if log == 0:
        #    print('total no train: {}'.format(sum(indices)))

        # Disagree
        indices = disagreement * ~current_win
        dynamic_weights[indices] = probabilities_current[indices] ** gamma2
        #if log == 0:
        #    print('total train: {}'.format(sum(indices)))
        # Weight loss
        total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()
        return total_loss, true_loss
    
    
class DynamicFlipLoss(_WeightedLoss):
    __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight, reduction)
        self.ignore_index = ignore_index
        self.criterion_ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')

    def forward(self, inputs, targets, gamma1, gamma2, margin=0, log=False):
        targets = targets.softmax(dim=1)
        real_targets = targets.argmax(1).clone().detach()
        total_loss = self.criterion_ce(inputs, real_targets)  # Pixel-level weight is always 1
        true_loss = total_loss.mean().item()

        outputs = inputs.softmax(dim=1).clone().detach()
        decision_current = outputs.argmax(1)  # N
        decision_pseudo = real_targets.clone().detach()  # N
        confidence_current = outputs.max(1).values  # N
        confidence_pseudo = targets.max(1).values.clone().detach()  # N added softmax otherwise pred >

        temp = decision_pseudo.unsqueeze(1).clone().detach()
        probabilities_current = outputs.gather(dim=1, index=temp).squeeze(1)

        dynamic_weights = torch.ones_like(decision_current).float()

        # Prepare indices
        disagreement = decision_current != decision_pseudo
        current_win = confidence_current > (confidence_pseudo + margin)

        # Agree
        indices = ~disagreement
        dynamic_weights[indices] = probabilities_current[indices] ** gamma1
        if log:
            print('total equal: {}'.format(sum(indices)))

        # Disagree (current model wins, do not learn!)
        indices = disagreement * current_win
        dynamic_weights[indices] = probabilities_current[indices] ** gamma2
        if log:
            print('total no train: {}'.format(sum(indices)))

        # Disagree
        indices = disagreement * ~current_win
        dynamic_weights[indices] = probabilities_current[indices] ** gamma2
        if log:
            print('total train: {}'.format(sum(indices)))

        # Weight loss
        total_loss = (total_loss * dynamic_weights).sum() / (real_targets != self.ignore_index).sum()
        return total_loss, true_loss
