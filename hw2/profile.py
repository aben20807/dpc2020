#!/usr/bin/env python
# -*- coding: utf-8 -*-
from resnet_cifar10 import ResNet20
import argparse
import chainer
from chainer import serializers
from chainer.function_hooks import TimerHook

import numpy as np


parser = argparse.ArgumentParser(description="Chainer example: Cifar-10")
parser.add_argument(
    "--out",
    "-o",
    default="../hw1/result/5/resnet20.model",
    help="Directory to output the result",
)
parser.add_argument(
    "--unit", "-u", type=int, default=10, help="Number of output layer units"
)
# parser.add_argument('--gpu', '-g', type=int, default=0,
# help='GPU ID (negative value indicates CPU)')#Set the initial matrixformat(numpy/cupy)
if __name__ == "__main__":
    args = parser.parse_args()  # The negative device number means CPU.
    # recorder.init()
    # print('GPU: {}'.format(args.gpu))
    # print("# unit: {}".format(args.unit))
    # print("")

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

    trials = 10000
    num_layer = 0  # For counting the number of layers of the model
    hook = TimerHook()
    y = []
    with hook:
        for i in range(trials):
            y = model(x)  # Inference result
            if i == 0:
                num_layer = len(hook.call_history)

    result = {}
    for i in range(len(hook.call_history)):
        layer_i = i % num_layer
        if not layer_i in result.keys():
            result[layer_i] = {
                "type": hook.call_history[i][0],
                "time": hook.call_history[i][1],
            }
        else:
            result[layer_i]["time"] += hook.call_history[i][1]
        i += 1

    average_time = dict(
        map(lambda kv: (kv[0], (kv[1]["time"] / trials)), result.items())
    )
    # hook.print_report()
    # import pprint
    # pprint.pprint(hook.call_history)
    # pprint.pprint(result)
    # pprint.pprint(average_time)
    for k in result.keys():
        print(f'{k}, {result[k]["type"]}, {average_time[k]}')

    # print("The test data label:", xp.asarray([test[0][1]]))
    # print("result:", y)
    # y_top5 = y.array[0].argsort()[-5:][::-1]
    # print("result Top 1:", [y_top5[0]])
    # print("result Top 5:", y_top5)