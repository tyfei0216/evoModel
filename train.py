import argparse
import json
import os

import numpy as np
import torch

import trainUtils


def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True)
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=[0])
    parser.add_argument("-s", "--strategy", type=str, default="auto")
    parser.add_argument("-n", "--name", type=str, default="ion_test")
    parser.add_argument("-c", "--checkpoint", type=str, default=None)
    args = parser.parse_args()
    return args


def run():
    args = parseArgs()
    path = args.path
    with open(os.path.join(path, "config.json"), "r") as f:
        configs = json.load(f)
    json_formatted_str = json.dumps(configs, indent=2)
    print("fetch config from ", os.path.join(path, "config.json"))
    print("--------")
    print("config: ")
    print(json_formatted_str)
    print("--------")
    print("using devices ", args.devices)
    print("using strategy ", args.strategy)
    print("--------")

    print("load pretrain model")
    pretrain_model = trainUtils.loadPretrainModel(configs)
    print("build finetune model")

    model = trainUtils.buildModel(configs, pretrain_model, args.checkpoint)

    print("load dataset")
    ds = trainUtils.loadDataset(configs)

    print("build trainer")
    trainer = trainUtils.buildTrainer(configs, args)

    print("start training")

    if args.checkpoint is not None:
        model.strict_loading = False
        trainer.fit(model, ds, ckpt_path=args.checkpoint)
    else:
        trainer.fit(model, ds)
    # torch.save(model.state_dict(), path + "parms.pt")


if __name__ == "__main__":
    run()
