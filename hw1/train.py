#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Reference:
#   1. https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
#   2. https://github.com/mitmul/chainer-cifar10/blob/master/models/ResNet.py
from __future__ import print_function
import chainer
from chainer import training
from chainer.training import extensions
from chainer import serializers
import chainer.links as L
import argparse
from resnet_cifar10 import ResNet20


parser = argparse.ArgumentParser(description="Chainer example: Cifar-10")
parser.add_argument(
    "--batchsize",
    "-b",
    type=int,
    default=128,
    help="Number of images in each mini-batch",
)
parser.add_argument(
    "--epoch",
    "-e",
    type=int,
    default=10,
    help="Number of sweeps over the dataset to train",
)
parser.add_argument(
    "--gpu", "-g", type=int, default=0, help="GPU ID (negative value indicates CPU)"
)  # Set the initial matrixformat(numpy/cupy)
parser.add_argument(
    "--out", "-o", default="result/4", help="Directory to output the result"
)
parser.add_argument(
    "--resume", "-r", default="", help="Resume the training from snapshot"
)
parser.add_argument(
    "--unit", "-u", type=int, default=10, help="Number of output layer units"
)

if __name__ == "__main__":
    args = parser.parse_args(["-g", "-1"])  # The negative device number means CPU.
    print("GPU: {}".format(args.gpu))
    print("# unit: {}".format(args.unit))
    print("# Minibatch-size: {}".format(args.batchsize))
    print("# epoch: {}".format(args.epoch))
    print("")

    model = ResNet20(args.unit)
    classifier_model = L.Classifier(model)
    # if args.gpu >= 0:
    # chainer.cuda.get_device(args.gpu).use() # Make a specified GPU current
    # classifier_model.to_gpu() # Copy the model to the GPU

    optimizer = (
        chainer.optimizers.Adam()
    )  # Adam is one of the Gradient descent optimizationalgorithms.
    optimizer.setup(classifier_model)

    train, test = chainer.datasets.get_cifar10()
    train_iter = chainer.iterators.SerialIterator(
        train, args.batchsize
    )  # Set Training data batchiterater
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False
    )
    # Forward the test data after each of training to calcuat the validation loss/arruracy.
    # updater and trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, (args.epoch, "epoch"), out=args.out)
    # extension objects
    trainer.extend(extensions.Evaluator(test_iter, classifier_model, device=args.gpu))
    trainer.extend(extensions.dump_graph("main/loss"))
    trainer.extend(extensions.snapshot(), trigger=(1, "epoch"))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(
            [
                "epoch",
                "main/loss",
                "validation/main/loss",
                "main/accuracy",
                "validation/main/accuracy",
                "elapsed_time",
            ]
        )
    )
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        serializers.load_npz(args.resume, trainer)
    trainer.run()
    serializers.save_npz(
        "{}/resnet20.model".format(args.out), model
    )  # Save the model(all trainedweights)