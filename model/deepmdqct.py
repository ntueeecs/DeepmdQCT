import os
import torch.nn as nn
import torch
import torchvision.models as models
from torchvision.utils import save_image
import transforms
import numpy as np
from torch.autograd import Function

BN_MOMENTUM = 0.5
model = models.resnet50()
mm = list(model.children())
mm[:1] = [nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)]
feature = mm[:-1]


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out=out+residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out=out+residual
        out = self.relu(out)

        return out


class StageModule(nn.Module):
    def __init__(self, input_branches, output_branches, c):
        """
        Build corresponding stages, which are used to integrate implementations of different scales
        :param input_ Branches: The number of input branches, each corresponding to a certain scale
        :param output_ Branches: Number of output branches
        : param c: Number of first branch channels input
        """
        super().__init__()
        self.input_branches = input_branches
        self.output_branches = output_branches

        self.branches = nn.ModuleList()
        for i in range(self.input_branches):
            w = c * (2 ** i)
            branch = nn.Sequential(
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w),
                BasicBlock(w, w)
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())
            for j in range(self.input_branches):
                if i == j:
                    self.fuse_layers[-1].append(nn.Identity())
                elif i < j:
                    self.fuse_layers[-1].append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=1, stride=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM),
                            nn.Upsample(scale_factor=2.0 ** (j - i), mode='nearest')
                        )
                    )
                else:  # i > j
                    ops = []
                    for k in range(i - j - 1):
                        ops.append(
                            nn.Sequential(
                                nn.Conv2d(c * (2 ** j), c * (2 ** j), kernel_size=3, stride=2, padding=1, bias=False),
                                nn.BatchNorm2d(c * (2 ** j), momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            )
                        )
                    ops.append(
                        nn.Sequential(
                            nn.Conv2d(c * (2 ** j), c * (2 ** i), kernel_size=3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2 ** i), momentum=BN_MOMENTUM)
                        )
                    )
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = [branch(xi) for branch, xi in zip(self.branches, x)]
        x_fused = []
        for i in range(len(self.fuse_layers)):
            x_fused.append(
                self.relu(
                    sum([self.fuse_layers[i][j](x[j]) for j in range(len(self.branches))])
                )
            )

        return x_fused


class up(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(up, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Sequential(nn.Conv2d(input_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.upsample(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Bottleneck_resnet(nn.Module):
    extention = 4

    def __init__(self, inplanes, planes, stride, downsample=None):
        super(Bottleneck_resnet, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.extention, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes * self.extention)

        self.relu = nn.ReLU()


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # 将残差部分和卷积部分相加
        out =out+residual
        out = self.relu(out)

        return out

class fusion(nn.Module):
    def __init__(self,input_channel,output_channel):
        super(fusion, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(input_channel,output_channel,
                                          kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(output_channel),nn.ReLU())
        self.conv2=nn.Sequential(nn.Conv2d(output_channel,output_channel,
                                          kernel_size=3, stride=1,padding=1),nn.BatchNorm2d(output_channel),nn.ReLU())
    def forward(self,x1,x2):
        x=torch.cat([x1,x2],dim=1)
        x=self.conv1(x)
        x =self.conv2(x)
        return x


    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DeepmdQCT(nn.Module):
    def make_layer(self, block, plane, block_num, stride=1):
        '''
        : param block: block template
        Param plane: The dimension of the intermediate operations in each module, usually equal to the output dimension/4
        :param block_ Num: number of repetitions
        : param stream: step size
        :return:
        '''
        block_list = []
        downsample = None
        if (stride != 1 or self.inplane != plane * block.extention):
            downsample = nn.Sequential(
                nn.Conv2d(self.inplane, plane * block.extention, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(plane * block.extention)
            )

        conv_block = block(self.inplane, plane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = plane * block.extention

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, plane, stride=1))

        return nn.Sequential(*block_list)

    def __init__(self, base_channel: int = 32, num_joints: int = 1):
        super().__init__()
        # Stem
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # Stage1
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM)
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64)
        )

        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base_channel, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(256, base_channel * 2, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 2, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage2
        self.stage2 = nn.Sequential(
            StageModule(input_branches=2, output_branches=2, c=base_channel)
        )

        # transition2
        self.transition2 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 4, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage3
        self.stage3 = nn.Sequential(
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel),
            StageModule(input_branches=3, output_branches=3, c=base_channel)
        )

        # transition3
        self.transition3 = nn.ModuleList([
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Identity(),  # None,  - Used in place of "None" because it is callable
            nn.Sequential(
                nn.Sequential(
                    nn.Conv2d(base_channel * 4, base_channel * 8, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(base_channel * 8, momentum=BN_MOMENTUM),
                    nn.ReLU(inplace=True)
                )
            )
        ])

        # Stage4
        # 注意，最后一个StageModule只输出分辨率最高的特征层
        self.stage4 = nn.Sequential(
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=4, c=base_channel),
            StageModule(input_branches=4, output_branches=1, c=base_channel)
        )

        # Final layer
        self.final_layer = nn.Conv2d(base_channel, num_joints, kernel_size=1, stride=1)

        # segement decoder
        self.up1 = up(192, 64)
        self.up2 = up(96, 32)
        self.up3 = up(96, 64)

        self.segment3 = nn.Sequential()
        self.segment3.add_module("up1", nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.segment3.add_module("bn4", nn.BatchNorm2d(64))
        self.segment3.add_module("relu4", nn.ReLU(inplace=True))

        self.segment3.add_module("conv5", nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
        self.segment3.add_module("bn5", nn.BatchNorm2d(64))
        self.segment3.add_module("relu5", nn.ReLU(inplace=True))

        self.segment3.add_module("conv7", nn.Conv2d(64, 6, kernel_size=3, stride=1, padding=1))
        self.segment3.add_module("sigmoid", nn.Sigmoid())

        # classification
        self.block = Bottleneck_resnet
        self.inplane = 64
        self.conv11 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage11 = self.make_layer(self.block, 64, 3, stride=1)
        self.stage22 = self.make_layer(self.block, 128, 4, stride=2)
        self.stage33 = self.make_layer(self.block, 256, 6, stride=2)
        self.stage44 = self.make_layer(self.block, 512, 3, stride=2)
        #feature fusion
        self.fusion1=fusion(256+64,256)
        self.fusion2=fusion(512+128,512)
        self.fusion3=fusion(1024+256,1024)

        self.avgpool = nn.AvgPool2d(8)
        # self.feature = nn.Sequential(*feature)
        self.class_classifier = nn.Sequential()   #类别分类
        self.class_classifier.add_module('c_fc1', nn.Linear(2048, 1024))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(1024))
        self.class_classifier.add_module('c_fc2', nn.Linear(1024, 3))
        self.class_classifier.add_module('c_soft', nn.Softmax())

        self.class_classifier_domian = nn.Sequential()   #domain cla
        self.class_classifier_domian.add_module('c_fc1', nn.Linear(2048, 1024))
        self.class_classifier_domian.add_module('c_bn1', nn.BatchNorm1d(1024))
        self.class_classifier_domian.add_module('c_fc2', nn.Linear(1024, 2))
        self.class_classifier_domian.add_module('c_soft', nn.Softmax())

        self.liner1 = nn.Linear(in_features=15, out_features=128)   #ct and clinic value  to qct
        self.liner2 = nn.Linear(in_features=128, out_features=256)
        self.liner3 = nn.Linear(in_features=256, out_features=1)




    def forward(self, x_s, targets,ct_and_clinic,alpha):

        flipped_images = transforms.flip_images(x_s)  # 翻转后的图像
        x = self.conv1(x_s)
        x = self.bn1(x)
        x1_ = self.relu(x)
        x = self.conv2(x1_)
        x = self.bn2(x)
        x2_ = self.relu(x)

        x = self.layer1(x2_)
        x = [trans(x) for trans in self.transition1]  # Since now, x is a list

        x = self.stage2(x)
        x3_ = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[-1])
        ]  # New branch derives from the "upper" branch only

        x = self.stage3(x3_)
        x_cla = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[-1]),
        ]  # New branch derives from the "upper" branch only

        x = self.stage4(x_cla)
        x1 = self.final_layer(x[0])  # loc

        # seg
        x2 = self.up1(x3_[2], x3_[1])
        x2 = self.up2(x2, x3_[0])
        x2 = self.up3(x2, x1_)
        x2 = self.segment3(x2)
        x2_1=x2[:,:3]  #L1 seg
        x2_2=x2[:,3:]  #L2  seg


        # keypoint
        m = self.conv1(flipped_images)
        m = self.bn1(m)
        m1_ = self.relu(m)
        m = self.conv2(m1_)
        m = self.bn2(m)
        m2_ = self.relu(m)

        m = self.layer1(m2_)
        m = [trans(m) for trans in self.transition1]  # Since now, x is a list

        m = self.stage2(m)
        m3_ = [
            self.transition2[0](m[0]),
            self.transition2[1](m[1]),
            self.transition2[2](m[-1])
        ]  # New branch derives from the "upper" branch only

        m = self.stage3(m3_)
        m = [
            self.transition3[0](m[0]),
            self.transition3[1](m[1]),
            self.transition3[2](m[2]),
            self.transition3[3](m[-1]),
        ]  # New branch derives from the "upper" branch only
        m = self.stage4(m)
        flipped_outputs = self.final_layer(m[0])  # loc

        flipped_outputs = transforms.flip_back(flipped_outputs, [[1, 1]])
        flipped_outputs[..., 1:] = flipped_outputs.clone()[..., 0:-1]
        outputs = (x1 + flipped_outputs) * 0.5
        reverse_trans = [torch.tensor(np.array([[5., -0., -63.875],
                                                [0., 5., -63.875]])) for t in targets]
        keypoints, scores = transforms.get_final_preds(outputs.detach().cpu(), reverse_trans, True)

        # label cla
        c1 = x2_1.clone()
        c2 = x2_2.clone()

        dis = 128
        result1 = torch.zeros((c1.shape[0], 3, 256, 256))
        result2 = torch.zeros((c2.shape[0], 3, 256, 256))
        c1 = torch.where(c1 * 255 > 1, x_s[:,0:3], c1)
        c2 = torch.where(c2 * 255 > 1, x_s[:,3:], c2)
        for m in range(x2_1.shape[0]):
            a, b = abs(int(keypoints[m][0][0])), abs(int(keypoints[m][0][1]))
            if a > dis and b > dis and a < (512 - dis) and b < (512 - dis):
                result1[m, :, :, :] = c1[m, :, a - dis:a + dis, b - dis:b + dis]
                result2[m, :, :, :] = c2[m, :, a - dis:a + dis, b - dis:b + dis]
            else:
                x = 256 - dis
                y = 256 + dis
                result1[m, :, :, :] = c1[m, :, x:y, x:y]
                result2[m, :, :, :] = c2[m, :, x:y, x:y]
        save_image(result1.cpu(), f"location_result_L1.png")
        save_image(result2.cpu(), f"location_result_L2.png")
        # cla myself
        # feature = self.feature(result.cuda())
        result=torch.cat([result1,result2],dim=1)
        out = self.conv11(result.cuda())
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # block
        out1 = self.stage11(out)
        out1 = self.fusion1(out1, x_cla[1])
        out2 = self.stage22(out1)
        out2 = self.fusion2(out2, x_cla[2])
        out3 = self.stage33(out2)
        out3 = self.fusion3(out3, x_cla[3])
        out4 = self.stage44(out3)

        #todo category cla
        feature = self.avgpool(out4)
        m7 = feature.view(feature.size(0), -1)
        x3 = self.class_classifier(m7)
        #todo domian cla
        reverse_feature = ReverseLayerF.apply(m7, alpha)
        domain_output = self.class_classifier_domian(reverse_feature)

        #todo Qct value fitting x3 probability ct_ and_ Clinic CT value and clinical value
        model_input= torch.cat([ct_and_clinic,x3],dim=1)
        qct = torch.relu(self.liner1(model_input))
        qct = torch.relu(self.liner2(qct))
        qct = self.liner3(qct)

        #todo X1->L1 and L2 positioning x2_ L1 vertebral body segmentation image x2_ 2->L2 vertebral segmentation image x3 category classification result domain_ Output Domain Classification Results
        return x1,x2_1,x2_2,x3,domain_output,qct




