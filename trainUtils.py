import os
import pickle

import numpy as np
import pytorch_lightning
import torch
import torch.nn as nn
from pytorch_lightning.loggers import TensorBoardLogger

# import pytorch_lightning as L
from pytorch_lightning.profilers import PyTorchProfiler
from torch.utils import tensorboard

import callbacks
import modules


def loadesm3(configs):
    from esm.models.esm3 import ESM3

    model = ESM3.from_pretrained("esm3_sm_open_v1").cpu()
    model = modules.fixParameters(
        model, unfix=configs["pretrain_model"]["unfix_layers"]
    )
    model = modules.addlora(
        model,
        layers=configs["pretrain_model"]["add_lora"],
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
    )
    return model


LOAD_PRETRAIN = {
    "esm3": loadesm3,
    # "esm3": loadesm3,
}


def loadPretrainModel(configs) -> nn.Module:
    # if "pretrain_model" in configs:
    #     model = "stage1"
    #     if "model" in configs["pretrain_model"]:
    #         model = "esm3"
    # else:
    #     return None

    if configs["pretrain_model"]["model"] in LOAD_PRETRAIN:
        return LOAD_PRETRAIN[configs["pretrain_model"]["model"]](configs)
    else:
        raise NotImplementedError


def loadPickle(path):
    with open(
        path,
        "rb",
    ) as f:

        name = os.path.basename(path)
        data = pickle.load(f)
        for j in data:
            j["origin"] = name
            if "strcture_t" in j:
                j["structure_t"] = j.pop("strcture_t")
    return data


def loadDatasetesm2(configs):
    import VirusDataset

    ds = VirusDataset.SeqdataModule(batch_size=configs["train"]["batch_size"])
    return ds


def loadDatasetesm3(configs):
    import VirusDataset

    data = loadPickle(configs["dataset"]["path"])

    step_points = configs["augmentation"]["step_points"]
    crop = configs["augmentation"]["crop"]
    maskp = [
        (i, j)
        for i, j in zip(
            configs["augmentation"]["maskp"], configs["augmentation"]["maskpc"]
        )
    ]
    aug = VirusDataset.DataAugmentation(
        step_points, maskp, crop, [], tracks=configs["dataset"]["tracks"]
    )

    # ds1 = VirusDataset.ESM3MultiTrackDataset(
    #     data1, data2, label, augment=aug, tracks=configs["dataset"]["tracks"]
    # )
    ds1 = VirusDataset.ESM3MultiTrackDatasetTEST(
        data, tracks=configs["dataset"]["tracks"], augment=aug
    )

    ds = VirusDataset.ESM3datamoduleSingle(ds1)
    return ds


def loadBalancedDatasetesm3ae(configs):
    import VirusDataset

    datasets = []
    lens = []

    sample_size = configs["dataset"]["dataset_train_sample"]
    assert len(configs["dataset"]["datasets"]) == len(sample_size)

    sample_size = configs["dataset"]["dataset_val_sample"]
    assert len(configs["dataset"]["datasets"]) == len(sample_size)

    for i in configs["dataset"]["datasets"]:
        data = loadPickle(i)
        datasets.append(data)

    step_points = configs["augmentation"]["step_points"]
    crop = configs["augmentation"]["crop"]
    maskp = configs["augmentation"]["maskp"]
    maskpc = configs["augmentation"]["maskpc"]
    mutate = configs["augmentation"]["mutate"]
    mutatep = configs["augmentation"]["mutatep"]
    tracks = configs["augmentation"]["tracks"]

    aug = VirusDataset.DataAugmentation(
        step_points, maskp, maskpc, crop, lens, mutate, mutatep, tracks=tracks
    )

    if "required_labels" not in configs["dataset"]:
        configs["dataset"]["required_labels"] = []

    ds = VirusDataset.ESM3BalancedDataModule(
        datasets,
        configs["train"]["batch_size"],
        sample_train=configs["dataset"]["dataset_train_sample"],
        sample_val=configs["dataset"]["dataset_val_sample"],
        train_test_ratio=configs["dataset"]["train_test_ratio"],
        aug=aug,
        tracks=configs["dataset"]["tracks"],
        required_labels=configs["dataset"]["required_labels"],
    )
    return ds


LOAD_DATASET = {
    "esm3ae": loadBalancedDatasetesm3ae,
}


def loadDataset(configs) -> pytorch_lightning.LightningDataModule:
    dataset = "esm2"
    if "dataset" in configs:
        dataset = configs["dataset"]["type"]

    if dataset in LOAD_DATASET:
        return LOAD_DATASET[dataset](configs)
    else:
        raise NotImplementedError


def buildAutoEncoder(configs, model):
    if "clf_params" not in configs["model"]:
        configs["model"]["clf_params"] = {}

    with open(configs["model"]["ori_seq"], "rb") as f:
        ori_seqs = pickle.load(f)

    ae = modules.AutoEncoder(
        model,
        lr=configs["model"]["lr"],
        weight_decay=configs["model"]["weight_decay"],
        classes=configs["model"]["classes"],
        clf_params=configs["model"]["clf_params"],
        masked_weight=configs["model"]["masked_weight"],
        label_weights=configs["model"]["label_weights"],
        ori_seqs=ori_seqs,
        l=configs["model"]["l"][0],
        tf=configs["model"]["teaching force"][0],
    )
    return ae


def buildFullModel(configs, model):
    raise NotImplementedError


BUILD_MODEL = {
    "stage1": buildAutoEncoder,
    "stage2": buildFullModel,
    # "esm3": buildesm3Model,
}


def buildModel(
    configs, basemodel=None, checkpoint=None
) -> pytorch_lightning.LightningModule:
    model = "stage1"
    if "type" in configs["model"]:
        model = configs["model"]["type"]

    if model in BUILD_MODEL:
        model = BUILD_MODEL[model](configs, basemodel)
    else:
        raise NotImplementedError

    if checkpoint is not None:
        t = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(t["state_dict"], strict=False)
        gs = t["global_step"]
        if "unfreeze" in configs["pretrain_model"]:
            t = configs["pretrain_model"]["unfreeze"]["steps"]
            idx = np.argsort(t)
            idx = filter(lambda x: t[x] < gs, idx)
            model.load_freeze = [
                configs["pretrain_model"]["unfreeze"]["layers"][i] for i in idx
            ]

    return model


def buildTrainer(configs, args):
    # k = 2
    # if "save" in configs["train"]:
    #     k = configs["train"]["save"]

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="validate_acc",  # Replace with your validation metric
    #     mode="max",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
    #     save_top_k=k,  # Save top k checkpoints based on the monitored metric
    #     save_last=True,  # Save the last checkpoint at the end of training
    #     dirpath=args.path,  # Directory where the checkpoints will be saved
    #     filename="{epoch}-{validate_acc:.2f}",  # Checkpoint file naming pattern
    # )

    cbs = callbacks.getCallbacks(configs, args)

    # profiler = PyTorchProfiler(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler(
    #         "tb_logs/%s" % args.name
    #     ),
    # )
    logger = TensorBoardLogger("tb_logs", name=args.name)

    if args.strategy == "deep":
        args.strategy = pytorch_lightning.strategies.DeepSpeedStrategy()

    pytorch_lightning.seed_everything(configs["train"]["seed"])

    trainer = pytorch_lightning.Trainer(
        strategy=args.strategy,
        logger=logger,
        accelerator="gpu",
        # profiler=profiler,
        devices=args.devices,
        max_epochs=configs["train"]["epoch"],
        log_every_n_steps=1,
        gradient_clip_val=configs["train"]["gradient_clip_val"],
        accumulate_grad_batches=configs["train"]["accumulate_grad_batches"],
        callbacks=cbs,
    )

    return trainer
