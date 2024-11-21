import torch
import torchvision

# ---- ENCODER

class Resnet(torch.nn.Module):
    def __init__(self, in_channels, n_classes=3, n_blocks=4, pretrained=False, mode='instance', aggregation='max',
                 weights="IMAGENET1K_V2", backbone="RN18"):
        super(Resnet, self).__init__()
        self.n_blocks = n_blocks
        self.n_classes = n_classes
        self.embedding = []
        self.pretrained = pretrained
        self.mode = mode  # 'embedding', 'instance'
        self.aggregation = aggregation  # 'max', 'mean'
        self.backbone = backbone

        if self.backbone == "RN50":
            print("Using Resnet-50 as backbone", end="\n")
            self.model = torchvision.models.resnet50(pretrained=weights)
            self.nfeats = 4*(512 // (2 ** (4 - n_blocks)))
        else:
            print("Using Resnet-18 as backbone", end="\n")
            self.model = torchvision.models.resnet18(pretrained="IMAGENET1K_V1")
            self.nfeats = 512 // (2 ** (4 - n_blocks))
        if in_channels != 3:
            self.input = torch.nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2),
                                         padding=(3, 3), bias=False)
        else:
            self.input = list(self.model.children())[0]
        self.classifier = torch.nn.Conv2d(in_channels=self.nfeats, out_channels=self.n_classes, kernel_size=(1, 1),
                                          bias=False)

        # placeholder for the gradients
        self.gradients = None

    def forward(self, x,reshape_cam=True):
        # Input dimensions
        _, _, H, W = x.size()

        # Input block - channels normalization
        x = self.input(x)
        for iBlock in range(1, 4):
            x = list(self.model.children())[iBlock](x)

        # Feature extraction
        F = []
        for iBlock in range(4, self.n_blocks+4):
            x = list(self.model.children())[iBlock](x)
            F.append(x)
        feature = x
        self.embedding = x
        # Output cams logits
        cam = self.classifier(x)
        if reshape_cam:
            cam = torch.nn.functional.interpolate(cam, size=(H, W), mode='bilinear', align_corners=True)

        # Image-level output
        if self.mode == 'instance':
            if self.aggregation == 'max':
                pred = torch.squeeze(torch.nn.AdaptiveMaxPool2d(1)(cam))
            if self.aggregation == 'mean':
                pred = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(cam))
        elif self.mode == 'embedding':
            if self.aggregation == 'max':
                embedding = torch.nn.AdaptiveMaxPool2d(1)(x)
            if self.aggregation == 'mean':
                embedding = torch.nn.AdaptiveMaxPool2d(1)(x)
            pred = torch.squeeze(self.classifier(embedding))

        return pred, feature,cam

class dua_network(torch.nn.Module):
    def __init__(self,weak_model,strong_model, n_classes=3, n_blocks=4, pretrained=False, mode='instance', aggregation='max',
                 weights="IMAGENET1K_V2", backbone="RN18"):
        super(dua_network, self).__init__()
        self.model1 = weak_model
        self.model2 = strong_model
        self.backbone = backbone
        self.n_classes = n_classes
        self.mode = mode
        self.aggregation = aggregation
        self.pretrained = pretrained
        if self.backbone == "RN50":
            print("Using Resnet-50 as backbone", end="\n")
            self.model = torchvision.models.resnet50(pretrained=weights)
            self.nfeats = 4 * (512 // (2 ** (4 - n_blocks)))
        else:
            print("Using Resnet-18 as backbone", end="\n")
            self.model = torchvision.models.resnet18(pretrained="IMAGENET1K_V1")
            self.nfeats = 512 // (2 ** (4 - n_blocks))

        self.classifier = torch.nn.Conv2d(in_channels=self.nfeats * 2, out_channels=self.n_classes, kernel_size=(1, 1),
                                          bias=False)

    def forward(self,x,y,reshape_cam=True):
        _, _, H, W = x.size()
        pred1,fea1,cam1  =self.model1(x)
        pred2,fea2,cam2 = self.model2(y)
        feature = torch.cat((fea1, fea2), dim=1)

        self.embedding = feature
        # Output cams logits
        cam = self.classifier(feature)
        if reshape_cam:
            cam = torch.nn.functional.interpolate(cam, size=(H, W), mode='bilinear', align_corners=True)

        # Image-level output
        if self.mode == 'instance':
            if self.aggregation == 'max':
                pred = torch.squeeze(torch.nn.AdaptiveMaxPool2d(1)(cam))
            if self.aggregation == 'mean':
                pred = torch.squeeze(torch.nn.AdaptiveAvgPool2d(1)(cam))
        elif self.mode == 'embedding':
            if self.aggregation == 'max':
                embedding = torch.nn.AdaptiveMaxPool2d(1)(x)
            if self.aggregation == 'mean':
                embedding = torch.nn.AdaptiveMaxPool2d(1)(x)
            pred = torch.squeeze(self.classifier(embedding))
        return pred1,pred2,pred,cam1,cam2,cam


if __name__ == '__main__':
    images = torch.rand(2, 3, 128,128).cuda(0)
    model1 = Resnet(in_channels=3)
    model1 = model1.cuda(0)
    model2 = Resnet(in_channels=3)
    # model2= model2.cuda(0)
    all_model = dua_network(model1,model2).cuda(0)
    print(all_model(images,images)[0].shape)
