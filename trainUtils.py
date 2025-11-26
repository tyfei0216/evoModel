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
import VirusDataset


def loadesm3(configs):
    from esm.models.esm3 import ESM3

    model = ESM3.from_pretrained("esm3_sm_open_v1", torch.device("cpu"))

    if "path" in configs["pretrain_model"]:
        a = torch.load(configs["pretrain_model"]["path"], "cpu")
        model.load_state_dict(a)
        print("load local model:",configs["pretrain_model"]["path"])

    unfixes = configs["pretrain_model"]["unfix_layers"]
    if (
        "unlock_norm_weights" in configs["pretrain_model"]
        and configs["pretrain_model"]["unlock_norm_weights"]
    ):
        unfixes.append("norm.weight")
        unfixes.append("layernorm_qkv.0.weight")
        unfixes.append("layernorm_qkv.0.bias")
        unfixes.append("q_ln.weight")
        unfixes.append("k_ln.weight")

    model = modules.fixParameters(model, unfix=unfixes)
    q = [
        "transformer.blocks." + str(s) + "."
        for s in configs["pretrain_model"]["add_lora"]
    ]
    model = modules.addlora(
        model,
        layers=q,
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
    )
    # model = modules.ESMModule(model, "esm3")
    return model


def loadNone(config):
    return None


def loadesmc(configs):
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(
        configs["pretrain_model"]["model"], torch.device("cpu")
    )  # .cpu()
    q = [
        "transformer.blocks." + str(s) + "."
        for s in configs["pretrain_model"]["add_lora"]
    ]
    unfixes = configs["pretrain_model"]["unfix_layers"]
    if (
        "unlock_norm_weights" in configs["pretrain_model"]
        and configs["pretrain_model"]["unlock_norm_weights"]
    ):
        unfixes.append("norm.weight")
        unfixes.append("layernorm_qkv.0.weight")
        unfixes.append("layernorm_qkv.0.bias")
        unfixes.append("q_ln.weight")
        unfixes.append("k_ln.weight")

    model = modules.fixParameters(
        model, unfix=configs["pretrain_model"]["unfix_layers"]
    )
    model = modules.addlora(
        model,
        layers=q,
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
        # dtype=torch.bfloat16,
    )
    # model = modules.ESMModule(model, "esmc")
    return model

def loadesm2(configs):
    import esm

    model, _ = esm.pretrained.esm2_t33_650M_UR50D()
    # model, _ = esm.pretrained.esm2_t12_35M_UR50D()

    unfixes = configs["pretrain_model"]["unfix_layers"]

    if (
        "unlock_norm_weights" in configs["pretrain_model"]
        and configs["pretrain_model"]["unlock_norm_weights"]
    ):
        unfixes.append("norm.weight")
        unfixes.append("norm.bias")

    model = modules.fixParameters(model, unfix=configs["pretrain_model"]["unfix_layers"])

    q = ["layers." + str(s) + "." for s in configs["pretrain_model"]["add_lora"]]

    model = modules.addlora(
        model,
        layers=q,
        ranks=configs["pretrain_model"]["rank"],
        alphas=configs["pretrain_model"]["alpha"],
    )

    return model

def loadDummy(configs):
    return modules.MyESM()


LOAD_PRETRAIN = {
    "esm3": loadesm3,
    "esmc_600m": loadesmc,
    "esmc_300m": loadesmc,
    "dummy":loadDummy,
    "esm2": loadesm2,
    "None": loadNone,
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


def loadDataset(configs) -> pytorch_lightning.LightningDataModule:
    # dataset = "esm2"
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

    aug = VirusDataset.DataAugmentation(
        step_points, maskp, maskpc, crop, lens, mutate, mutatep
    )

    if "required_labels" not in configs["dataset"]:
        configs["dataset"]["required_labels"] = []
        
    if "all_for_train" not in configs["dataset"]:
        configs["dataset"]["all_for_train"] = False

    # train_from_emb = configs["dataset"].get("train_from_emb", False)
    
    val_data = None
    # 检查config中是否有指定验证集
    if "val_dataset_path" in configs["dataset"]:
        val_path = configs["dataset"]["val_dataset_path"]
        print(f"正在加载手动指定的验证集: {val_path}")
        with open(val_path, "rb") as f:
            val_data = pickle.load(f)

    ds = VirusDataset.VESMDataModule(
        datasets,
        seq=configs["dataset"]["seq"],
        stage=configs["model"]["stage"],
        batch_size=configs["train"]["batch_size"],
        sample_train=configs["dataset"]["dataset_train_sample"],
        sample_val=configs["dataset"]["dataset_val_sample"],
        train_test_ratio=configs["dataset"]["train_test_ratio"],
        aug=aug,
        stage_2_maskp=configs["augmentation"]["stage_2_maskp"],
        required_labels=configs["dataset"]["required_labels"],
        train_time_series=configs["dataset"]["train_time_series"],
        all_for_train=configs["dataset"]["all_for_train"],
        # train_from_emb=train_from_emb
        val_data=val_data
    )
    return ds


def buildModel(
    configs, basemodel=None, checkpoint=None
) -> pytorch_lightning.LightningModule:
    model = "stage1"
    if "type" in configs["model"]:
        model = configs["model"]["type"]

    # if "ori_seq" in configs["model"]["params"]:
    #     with open(configs["model"]["ori_seq"], "rb") as f:
    #         ori_seq = pickle.load(f)

    #     configs["model"]["params"]["ori_seqs"] = ori_seq
    # else:
    #     configs["model"]["params"]["ori_seqs"] = {}

    config = modules.VESMConfig(**configs["model"]["params"])
    stage = configs["model"]["stage"]
    model = modules.VESM(basemodel, stage, config)
    # if "checkpoint" in configs["model"] and configs["model"]["checkpoint"]:
    #     print("load model from checkpoint", configs["model"]["checkpoint"])
    #     model.load_state_dict(
    #         torch.load(configs["model"]["checkpoint"])["state_dict"], strict=False
    #     )

    if "checkpoint" in configs["model"] and configs["model"]["checkpoint"] is not None:
        t = torch.load(configs["model"]["checkpoint"], map_location="cpu")
        ckpt = t["state_dict"]
        need_del = []
        p = list(model.state_dict().keys())  # [j for j, _ in model.named_parameters()]
        # print("model parameters ", p)
        for i in ckpt:
            if i not in p or ckpt[i].shape != model.state_dict()[i].shape:
                need_del.append(i)
        for i in need_del:
            del ckpt[i]
        print("incompatible parameters", need_del)
        model.load_state_dict(ckpt, strict=False)
        print("finish loading parameters")
        
    if checkpoint is not None:
        print("load model from checkpoint", checkpoint)
        t = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(t["state_dict"], strict=False)
    
    if model.stage == "training stage 2":
        for i, j in model.named_parameters():
            if "esm_model" in i:
                j.requires_grad_(False)       
    return model


# 使用SimpleESM，只针对综述
def buildModelS(
    configs, basemodel=None, checkpoint=None
) -> pytorch_lightning.LightningModule:
    # model = "stage1"
    # if "type" in configs["model"]:
    #     model = configs["model"]["type"]

    # if "ori_seq" in configs["model"]["params"]:
    #     with open(configs["model"]["ori_seq"], "rb") as f:
    #         ori_seq = pickle.load(f)

    #     configs["model"]["params"]["ori_seqs"] = ori_seq
    # else:
    #     configs["model"]["params"]["ori_seqs"] = {}

    config = modules.VESMConfig(**configs["model"]["params"])
    # stage = configs["model"]["stage"]
    model = modules.SimpleESM(basemodel, config)
    if "checkpoint" in configs["model"] and configs["model"]["checkpoint"]:
        print("load model from checkpoint", configs["model"]["checkpoint"])
        model.load_state_dict(
            torch.load(configs["model"]["checkpoint"])["state_dict"], strict=False
        )
    if checkpoint is not None:
        print("load model from checkpoint", checkpoint)
        t = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(t["state_dict"], strict=False)

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
    logger = TensorBoardLogger(configs["train"]["logger_path"], name=configs["train"]["name"])

    if args.strategy == "deep":
        args.strategy = pytorch_lightning.strategies.DeepSpeedStrategy()

    pytorch_lightning.seed_everything(configs["train"]["seed"])

    if "val_check_interval" not in configs["train"]:
        configs["train"]["val_check_interval"] = None

    if "gradient_clip_val" not in configs["train"]:
        configs["train"]["gradient_clip_val"] = None

    trainer = pytorch_lightning.Trainer(
        strategy=args.strategy,
        logger=logger,
        accelerator="gpu",
        # profiler=profiler,
        devices=args.devices,
        max_epochs=configs["train"]["epoch"],
        log_every_n_steps=1,
        val_check_interval=configs["train"]["val_check_interval"],
        gradient_clip_val=configs["train"]["gradient_clip_val"],
        accumulate_grad_batches=configs["train"]["accumulate_grad_batches"],
        callbacks=cbs,

        
        precision='bf16-mixed'
    )

    return trainer
