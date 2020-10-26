#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference:
#   1. https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
#   2. https://github.com/mitmul/chainer-cifar10/blob/master/models/ResNet.py
import chainer
import chainer.functions as F
import chainer.links as L


class BottleNeck(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1):
        self.shortcut = stride != 1
        super(BottleNeck, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(
                n_in, n_out, ksize=3, stride=stride, pad=1, nobias=True
            )
            self.bn1 = L.BatchNormalization(n_out)
            self.conv2 = L.Convolution2D(
                n_out, n_out, ksize=3, stride=1, pad=1, nobias=True
            )
            self.bn2 = L.BatchNormalization(n_out)
            self.conv3 = L.Convolution2D(
                n_in, n_out, ksize=1, stride=stride, pad=0, nobias=True
            )
            self.bn3 = L.BatchNormalization(n_out)

    def __call__(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        if self.shortcut:
            h += self.bn3(self.conv3(x))
        h = F.relu(h)
        return h


class Block(chainer.ChainList):
    def __init__(self, n_in, n_out, n_bottlenecks, stride):
        super(Block, self).__init__()
        self.in_planes = n_in
        strides = [stride] + [1] * (n_bottlenecks - 1)
        for stride in strides:
            self.add_link(BottleNeck(self.in_planes, n_out, stride))
            self.in_planes = n_out

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x


class ResNet(chainer.Chain):
    def __init__(self, n_class=10, n_blocks=[3, 3, 3]):
        super(ResNet, self).__init__()

        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 16, 3, 1, 1, nobias=True)
            self.bn2 = L.BatchNormalization(16)
            self.res3 = Block(16, 16, n_blocks[0], 1)
            self.res4 = Block(16, 32, n_blocks[1], 2)
            self.res5 = Block(32, 64, n_blocks[2], 2)
            self.fc7 = L.Linear(64, n_class)

    def __call__(self, x):
        h = F.relu(self.bn2(self.conv1(x)))
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = F.average_pooling_2d(h, h.shape[2:])
        h = self.fc7(h)
        return h


class ResNet20(ResNet):
    def __init__(self, n_class=10):
        super(ResNet20, self).__init__(n_class, [3, 3, 3])
