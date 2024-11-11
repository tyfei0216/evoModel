from typing import Any, List, Mapping

import pytorch_lightning as L
import torch
import torchmetrics
from pytorch_lightning.callbacks import BaseFinetuning, ModelCheckpoint
from torch.optim import Optimizer
from torch.utils import tensorboard


class LambdaUpdate(L.Callback):
    def __init__(self, warmup=5000, check=3000):
        self.warmup = warmup
        self.check = check
        self.acc = torchmetrics.Accuracy(task="binary")

        self.cnt = 0
        self.outputs = []

    def on_train_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: torch.Tensor | Mapping[str, Any] | None,
        batch: torch.Any,
        batch_idx: int,
    ) -> None:
        self.outputs.append(pl_module.training_step_outputs[-1])
        # return super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer
    ) -> None:
        if self.warmup > 0:
            self.warmup -= 1
        else:
            self.cnt += 1

        if self.cnt == self.check:
            self.cnt = 0
            scores = torch.concatenate([x["y"] for x in self.outputs])
            y = torch.concatenate([x["true_label"] for x in self.outputs])

            self.outputs.clear()
            acc = self.acc(scores, y)
            pl_module.updateLambda(acc)
        # return super().on_before_optimizer_step(trainer, pl_module, optimizer)


class DatasetAugmentationUpdate(L.Callback):
    def __init__(self):
        pass

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer
    ) -> None:
        trainer.train_dataloader.step()
        # print(trainer.train_dataloader)
        # for i in trainer.train_dataloader:
        #     i.step()


class FinetuneUpdates(BaseFinetuning):
    def __init__(self, iters=[], unfreeze_layers=[]):
        super().__init__()
        self.iters = iters
        self.layers = unfreeze_layers
        assert len(self.iters) == len(self.layers)
        self.cnt = 0
        self.update = True

    def on_before_optimizer_step(
        self, trainer: L.Trainer, pl_module: L.LightningModule, optimizer: Optimizer
    ) -> None:
        self.update = False
        self.cnt += 1

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        # if current_epoch == self._unfreeze_at_epoch:
        #     self.unfreeze_and_add_param_group(
        #         modules=pl_module.feature_extractor,
        #         optimizer=optimizer,
        #         train_bn=True,
        #     )
        pass

    def on_train_batch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule, batch, batch_idx
    ) -> None:
        if self.update:
            return
        self.update = True
        update = []
        unfreeze = []
        for i, layers in zip(self.iters, self.layers):
            if i == self.cnt:
                for j, k in pl_module.named_modules():
                    flag = 1
                    for l in layers:
                        if l not in j:
                            flag = 0
                            break
                    if flag == 1:
                        update.append(k)
                        unfreeze.append(j)
        if len(unfreeze) != 0:
            print("reached target iteration %i" % self.cnt)
            print("unfreezing ", unfreeze)
            self.unfreeze_and_add_param_group(
                modules=update,
                optimizer=trainer.optimizers[0],
            )
            # module_path = "esm_model.transformer.blocks.47.ffn.3.lora"
            # submodule = pl_module
            # tokens = module_path.split(".")
            # for token in tokens:
            #     submodule = getattr(submodule, token)
            # print(submodule.B)

    def freeze_before_training(self, pl_module: L.LightningModule) -> None:
        update = []
        freeze = []
        for layers in self.layers:
            for j, k in pl_module.named_modules():
                flag = 1
                for l in layers:
                    if l not in j:
                        flag = 0
                        break
                if flag == 1:
                    update.append(k)
                    freeze.append(j)

        if len(freeze) != 0:
            print("starting training and freeze modules ", freeze)
            self.freeze(update)


def getCallbacks(configs, args) -> List[L.Callback]:

    ret = []
    k = 2
    if "save" in configs["train"]:
        k = configs["train"]["save"]

    checkpoint_callback = ModelCheckpoint(
        monitor="validate_loss",  # Replace with your validation metric
        mode="min",  # 'min' if the metric should be minimized (e.g., loss), 'max' for maximization (e.g., accuracy)
        save_top_k=k,  # Save top k checkpoints based on the monitored metric
        save_last=True,  # Save the last checkpoint at the end of training
        dirpath=args.path,  # Directory where the checkpoints will be saved
        filename="{epoch}-{validate_loss:.2f}",  # Checkpoint file naming pattern
    )
    ret.append(checkpoint_callback)

    if "augmentation" in configs:
        print("build data augmentation callback")
        au = DatasetAugmentationUpdate()
        ret.append(au)

    if "pretrain_model" in configs:
        print("build pretrain model unfreeze callback")
        if "unfreeze" in configs["pretrain_model"]:
            if args.checkpoint is None:
                ft = FinetuneUpdates(
                    iters=configs["pretrain_model"]["unfreeze"]["steps"],
                    unfreeze_layers=configs["pretrain_model"]["unfreeze"]["layers"],
                )
                ret.append(ft)

    return ret
