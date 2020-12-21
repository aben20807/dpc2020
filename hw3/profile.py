#!/usr/bin/env python
# -*- coding: utf-8 -*-
from resnet_cifar10 import ResNet20
import argparse
import chainer
from chainer import serializers
from chainer.function_hooks import TimerHook
from tqdm import tqdm

import numpy as np
import cupy
import common


parser = argparse.ArgumentParser(description="Chainer example: Cifar-10")
parser.add_argument(
    "--weights",
    "-w",
    default="../hw1/result/5/resnet20.model",
    help="Path of the trained weights",
)
parser.add_argument(
    "--unit", "-u", type=int, default=10, help="Number of output layer units"
)
parser.add_argument(
    "--mode",
    "-m",
    type=int,
    default=0,
    help="0: All CPU; 1: All GPU; 2: First Conv for each Block on GPU",
)
parser.add_argument(
    "--profile", "-p", type=str, default="time", choices=["time", "memory"]
)
parser.add_argument(
    "--output", "-o", type=str, default="./log.txt", help="Log path for output"
)
# parser.add_argument('--gpu', '-g', type=int, default=0,
# help='GPU ID (negative value indicates CPU)')#Set the initial matrixformat(numpy/cupy)
if __name__ == "__main__":
    args = parser.parse_args()  # The negative device number means CPU.

    # Load the dataset
    _, test = chainer.datasets.get_cifar10()

    xp = np
    common.FIRST_CONV_GPU_FLAG = False
    if args.mode == 2:
        common.FIRST_CONV_GPU_FLAG = True

    model = ResNet20(args.unit)
    if args.mode == 0:
        pass
    elif args.mode == 1:
        model.to_gpu()  # Copy the model to the GPU
        xp = cupy

    # Load trained model
    serializers.load_npz(args.weights, model)

    x = chainer.Variable(xp.asarray([test[0][0]]))  # test data
    t = chainer.Variable(xp.asarray([test[0][1]]))  # labels

    trials = 10000
    num_layer = 0  # For counting the number of layers of the model
    pbar = tqdm(total=trials)
    if args.profile == "time":
        hook = TimerHook()
        y = []
        with hook:
            for i in range(trials):
                y = model(x)  # Inference result
                if i == 0:
                    num_layer = len(hook.call_history)
                pbar.update()
            pbar.close()

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
        with open(args.output, "w") as fout:
            for k in result.keys():
                fout.write(f'{k}, {result[k]["type"]}, {average_time[k]}\n')

    elif args.profile == "memory":
        from cupy.cuda import memory_hooks

        mem_hook = memory_hooks.LineProfileHook(max_depth=0)
        y = []
        with mem_hook:
            for i in range(trials):
                y = model(x)  # Inference result
                pbar.update()
            pbar.close()
        with open(args.output, "w") as fout:
            mem_hook.print_report(file=fout)
