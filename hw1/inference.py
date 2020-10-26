#!/usr/bin/env python
# -*- coding: utf-8 -*-
from resnet_cifar10 import ResNet20
import argparse
import chainer
from chainer import serializers
import numpy as np

parser = argparse.ArgumentParser(description="Chainer example: Cifar-10")
parser.add_argument(
    "--out", "-o", default="result/4/resnet20.model", help="Directory to output the result"
)
parser.add_argument(
    "--unit", "-u", type=int, default=10, help="Number of output layer units"
)
# parser.add_argument('--gpu', '-g', type=int, default=0,
# help='GPU ID (negative value indicates CPU)')#Set the initial matrixformat(numpy/cupy)
if __name__ == "__main__":
    args = parser.parse_args()  # The negative device number means CPU.
    # print('GPU: {}'.format(args.gpu))
    print("# unit: {}".format(args.unit))
    print("")

    # Load the dataset
    _, test = chainer.datasets.get_cifar10()
    # Load trained model
    model = ResNet20(args.unit)
    # if args.gpu >= 0:
    # chainer.cuda.get_device(args.gpu).use() # Make a specified GPU current
    # model.to_gpu() # Copy the model to the GPU
    # xp = np if args.gpu < 0 else chainer.cuda.cupy
    xp = np

    serializers.load_npz(args.out, model)

    x = chainer.Variable(xp.asarray([test[0][0]]))  # test data
    t = chainer.Variable(xp.asarray([test[0][1]]))  # labels
    y = model(x)  # Inference result
    print("The test data label:", xp.asarray([test[0][1]]))
    print("result:", y)
    y_top5 = y.array[0].argsort()[-5:][::-1]
    print("result Top 1:", [y_top5[0]])
    print("result Top 5:", y_top5)