import random

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune
from attr import dataclass


def fixParameters(esm_model, unfix=["9", "10", "11"]):
    for i, j in esm_model.named_parameters():
        flag = 1
        for k in unfix:
            if k in i:
                flag = 0

        if flag == 1:
            j.requires_grad = False
        else:
            j.requires_grad = True

    return esm_model


class SelfAttention(nn.Module):
    def __init__(self, channels, n_head):
        super(SelfAttention, self).__init__()
        self.channels = channels
        # self.size = size
        self.n_head = n_head
        assert channels % n_head == 0
        self.rope = torchtune.modules.RotaryPositionalEmbeddings(channels // n_head)
        self.mha = nn.MultiheadAttention(channels, n_head, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x = x.swapaxes(1, 2)
        batch, length, channel = x.shape
        x = x.view(batch, length, self.n_head, self.channels // self.n_head)
        # print(x.shape)
        # print(self.rope(x).shape)
        x = self.rope(x)
        # print(x.shape)
        x = x.view(batch, length, self.channels)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value  # .swapaxes(2, 1)


class DecoderBlock(nn.Module):
    def __init__(self, channels, n_head, classes, transformer_layers=3):
        super(DecoderBlock, self).__init__()
        self.channels = channels
        self.n_head = n_head
        self.classes = classes

        self.transformer_blocks = nn.ModuleList(
            [SelfAttention(channels, n_head) for i in range(transformer_layers)]
        )
        # self.T1 = SelfAttention(channels, n_head)
        # self.T2 = SelfAttention(channels, n_head)
        # self.T3 = SelfAttention(channels, n_head)
        self.clf = nn.Linear(channels, classes)

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        x = self.clf(x)
        return x


class Linearcls(nn.Module):
    """simple linear classifier

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self, input_dim=256, take_embed="first", dropout=-1, p0=None, output_dim=1
    ):
        super().__init__()

        assert take_embed in ["first", "mean", "max", "last"]
        self.embed_dim = input_dim
        self.dropout = dropout
        self.take_embed = take_embed

        self.l1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.l2 = nn.Linear(self.embed_dim // 2, self.embed_dim // 4)
        self.l3 = nn.Linear(self.embed_dim // 4, output_dim)
        self.ln1 = nn.LayerNorm(self.embed_dim // 2)
        self.ln2 = nn.LayerNorm(self.embed_dim // 4)
        if p0 is None:
            self.p0 = None
        else:
            self.p0 = nn.Dropout(p0)
        if dropout > 0 and dropout < 1:
            self.dropout1 = nn.Dropout(p=self.dropout)
            self.dropout2 = nn.Dropout(p=self.dropout)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x: torch.Tensor):

        if self.take_embed == "first":
            x = x[:, 0]
        elif self.take_embed == "mean":
            x = torch.mean(x, dim=1)
        elif self.take_embed == "max":
            x = x.transpose(1, 2)
            x = F.adaptive_max_pool1d(x, 1)
        elif self.take_embed == "last":
            x = x[:, -1]
        else:
            raise NotImplementedError

        if self.p0 is not None:
            x = self.p0(x)

        x = self.l1(x)
        x = self.ln1(x)
        if self.dropout1 is not None:
            x = self.dropout1(x)
        x = F.gelu(x)
        x = self.l2(x)
        x = self.ln2(x)
        if self.dropout2 is not None:
            x = self.dropout2(x)
        x = F.gelu(x)
        x = self.l3(x)
        # print("lin", x.shape)
        return x


class CrossGeneModel(L.LightningModule):
    def __init__(
        self,
        in_channels=256,
        transformer_layers=5,
        n_head=16,
        seq=["NSP5", "E", "S"],
        weight_decay=0.0,
        masked_weight=0.1,
        lr=1e-4,
        clf_params={},
        only_embed=False,
        label_weights=None,
        l=1.0,
        classes=33,
        ori_seqs={},
    ):
        super().__init__()

        self.seq = seq
        self.weight_decay = weight_decay
        self.lr = lr

        # print(in_channels, n_head)

        self.encoder_blocks = nn.ModuleList(
            [SelfAttention(in_channels, n_head) for i in range(transformer_layers)]
        )

        self.decoder_block = nn.ModuleList(
            [SelfAttention(in_channels, n_head) for i in range(transformer_layers)]
        )
        self.cri = nn.MSELoss()
        if label_weights is not None:
            self.cri2 = nn.BCEWithLogitsLoss(weight=torch.tensor(label_weights))
        else:
            self.cri2 = nn.BCEWithLogitsLoss()
        self.clf = Linearcls(**clf_params)
        self.only_embed = only_embed

        self.ori_seqs = {}
        for i in ori_seqs:
            t = torch.tensor(ori_seqs[i], requires_grad=False)
            self.ori_seqs[i] = t
            self.register_buffer("ori_seq_" + i, t)
        self.embed = nn.Embedding(classes, in_channels)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.l = l
        self.masked_weight = masked_weight

    def forward(self, x):
        inputs = []
        for i in self.seq:
            if x[i].dim() == 1:
                x[i] = x[i].unsqueeze(0)
            inputs.append(x[i])

        inputs.append(torch.zeros_like(inputs[-1]))

        inputs = torch.stack(inputs, dim=1)

        for block in self.encoder_blocks:
            inputs = block(inputs)

        embed = inputs[:, -1, :]

        clsres = self.clf(inputs)

        if self.only_embed:
            return embed

        x = embed[:, None].repeat(1, len(self.seq), 1)
        # print(x.shape)
        for block in self.decoder_block:
            x = block(x)
        # print(x.shape)
        return embed, x, clsres

    def _common_training_step(self, input_dict, y, mask, labels=None):
        if isinstance(labels, list):
            labels = labels[0]
        self.only_embed = False
        _, x, res = self.forward(input_dict)
        # print(x.shape, y.shape)
        x = x.flatten()
        y = y.flatten()
        mask = mask.flatten()
        mask = 1 - mask
        mask[mask < 0] = -self.masked_weight
        mask += self.masked_weight
        x = x - y
        x = x * mask
        loss1 = self.cri(x, torch.zeros_like(x))

        if labels.shape[1] != 0:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            loss2 = self.cri2(res, labels)
        else:
            loss2 = 0.0
        loss = loss1 + self.l * loss2
        # print(loss.shape, loss)
        # exit()
        return loss, loss1, loss2

    def _parseBatch(self, batch):
        input_dict, mask, labels = batch
        y = []
        for i in self.seq:
            if input_dict["ori_" + i].dim() == 1:
                input_dict["ori_" + i] = input_dict["ori_" + i].unsqueeze(0)
            y.append(input_dict["ori_" + i])
        y = torch.stack(y, dim=1)

        masks = []
        for i in self.seq:
            if mask[i].dim() == 1:
                mask[i] = mask[i].unsqueeze(0)
            masks.append(mask[i])

        masks = torch.stack(masks, dim=1)

        return input_dict, y, masks, labels

    def training_step(self, batch, batch_idx):

        input_dict, y, masks, labels = self._parseBatch(batch)

        # mask = input_dict["mask"]
        loss, loss1, loss2 = self._common_training_step(input_dict, y, masks, labels)
        if not isinstance(loss2, float):
            loss2 = loss2.detach().cpu()
        else:
            loss2 = torch.tensor([0.0])
        self.training_step_outputs.append(
            {
                "total loss": loss.detach().cpu(),
                "predict loss": loss2,
                "reconstruct loss": loss1.detach().cpu(),
            }
        )
        self.log("train_loss:", loss, prog_bar=True)
        self.log("predict loss:", loss2, prog_bar=False)
        self.log("reconstruct loss:", loss1, prog_bar=False)
        return loss

    def _common_epoch_end(self, outputs):

        if len(outputs) == 0:
            return 0, 0, 0

        loss = torch.stack([i["total loss"] for i in outputs]).mean()
        loss1 = torch.stack([i["reconstruct loss"] for i in outputs]).mean()
        loss2 = torch.stack([i["predict loss"] for i in outputs]).mean()
        outputs.clear()
        # print(loss, loss1, loss2)
        return loss, loss1, loss2

    def on_training_epoch_end(self):

        loss, loss1, loss2 = self._common_epoch_end(self.training_step_outputs)

        print("finish training epoch, loss %f" % loss)
        # self.log_dict(
        #     {
        #         "epoch_train_loss": loss,
        #     },
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

    def on_validation_epoch_end(self):
        loss, loss1, loss2 = self._common_epoch_end(self.validation_step_outputs)
        print("finish validating, loss %f" % (loss))
        self.log_dict(
            {
                "epoch_validate_loss": loss1 + loss2,
                "epoch_validate_reconstruct_loss": loss1,
                "epoch_validate_predict_loss": loss2,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def validation_step(self, batch, batch_idx):

        input_dict, y, masks, labels = self._parseBatch(batch)

        loss, loss1, loss2 = self._common_training_step(input_dict, y, masks, labels)
        if not isinstance(loss2, float):
            loss2 = loss2.detach().cpu()
        else:
            loss2 = torch.tensor([0.0])
        self.validation_step_outputs.append(
            {
                "total loss": loss.detach().cpu(),
                "predict loss": loss2,
                "reconstruct loss": loss1.detach().cpu(),
            }
        )

        return loss

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i in checkpoint["state_dict"]:
            if "esm" in i and "lora" not in i:
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]

    def configure_optimizers(self):

        print("get training optimizer")
        if self.load_freeze is None:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            l1 = []
            for i, j in self.named_parameters():
                if "esm" not in j:
                    l1.append(j)
            optimizer = torch.optim.Adam(
                l1,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            for need in self.load_freeze:
                params = []
                for i, j in self.named_parameters():
                    flag = 1
                    for k in need:
                        if k not in i:
                            flag = 0
                            break
                    if flag == 1:
                        params.append(j)
                optimizer.add_param_group({"params": params, "lr": self.lr})

        return optimizer


class AutoEncoder(L.LightningModule):
    def __init__(
        self,
        esm_model,
        in_channels=1536,
        out_channels=256,
        n_head=16,
        lr=1e-4,
        lr_backbone=1e-5,
        only_embed=True,
        weight_decay=0.0,
        classes=33,
        clf_params={},
        label_weights=None,
        masked_weight=0.1,
        l=1.0,
        tf=0.5,
        ori_seqs={},
        transformer_layers=3,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["esm_model"])
        self.esm_model = esm_model
        self.bottleneck = nn.Linear(in_channels, out_channels)

        self.decoder = DecoderBlock(
            out_channels, n_head, classes, transformer_layers=transformer_layers
        )
        self.clf = Linearcls(**clf_params)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_head = n_head
        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay
        self.cri = nn.CrossEntropyLoss(reduction="none")

        self.cri2 = nn.BCEWithLogitsLoss(weight=torch.tensor(label_weights))
        self.only_embed = only_embed
        self.masked_weight = masked_weight

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.l = l
        self.tf = tf
        self.ori_seqs = {}
        for i in ori_seqs:
            t = torch.tensor(ori_seqs[i], requires_grad=False)
            self.ori_seqs[i] = t
            self.register_buffer("ori_seq_" + i, t)
        self.embed = nn.Embedding(classes, out_channels)

    def forward(self, input_dict):
        prot = input_dict["prot"]
        # print(prot)
        for i in ["seq_t", "structure_t", "ss8_t", "sasa_t"]:
            if i not in input_dict:
                input_dict[i] = None
            else:
                if len(input_dict[i].size()) == 1:
                    input_dict[i] = input_dict[i].unsqueeze(0)

        representations = self.esm_model(
            sequence_tokens=input_dict["seq_t"],
            structure_tokens=input_dict["structure_t"],
            ss8_tokens=input_dict["ss8_t"],
            sasa_tokens=input_dict["sasa_t"],
        )

        x = representations.embeddings

        batchsize, length, channels = x.shape

        embed = self.bottleneck(x)

        res = self.clf(embed)

        embed = embed[:, 0]

        if self.only_embed:
            return embed

        x = embed[:, None, :].repeat(1, length, 1)
        if self.tf > 0 and random.random() < self.tf:
            q = []
            for i in prot:
                q.append(getattr(self, "ori_seq_" + i))
            q = torch.stack(q)
            q = self.embed(q)
            if q.dim() == 2:
                q = q.unsqueeze(0)
            # print(q.shape, x.shape)
            l = min(x.shape[1], q.shape[1])
            x[:, :l, :] += q[:, :l, :]

        # print(x.shape)

        x = self.decoder(x)
        return embed, x, res

    def _common_training_step(self, input_dict, y, mask, labels=None):
        if isinstance(labels, list):
            labels = labels[0]
        self.only_embed = False
        _, x, res = self.forward(input_dict)
        x = x.view(-1, x.shape[-1])
        y = y.flatten()
        mask = mask.flatten()
        mask = 1 - mask
        mask[mask < 0] = -self.masked_weight
        mask += self.masked_weight
        loss = self.cri(x, y)
        if labels is not None:
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            loss2 = self.cri2(res, labels)
        else:
            loss2 = 0
        # print(loss.shape, mask.shape)
        loss = loss * mask
        # print(loss.shape)
        loss1 = loss.sum() / mask.sum()
        loss = loss1 + self.l * loss2
        # print(loss.shape, loss)
        # exit()
        return loss, loss1, loss2

    def training_step(self, batch, batch_idx):
        input_dict, labels = batch
        y = input_dict["ori_seq_t"]
        mask = input_dict["mask"].float()
        loss, loss1, loss2 = self._common_training_step(input_dict, y, mask, labels)
        self.training_step_outputs.append(
            {
                "total loss": loss.detach().cpu(),
                "predict loss": loss2.detach().cpu(),
                "reconstruct loss": loss1.detach().cpu(),
            }
        )
        self.log("train_loss:", loss, prog_bar=True)
        self.log("predict loss:", loss2, prog_bar=False)
        self.log("reconstruct loss:", loss1, prog_bar=False)
        return loss

    def _common_epoch_end(self, outputs):

        if len(outputs) == 0:

            return 0, 0, 0

        loss = torch.stack([i["total loss"] for i in outputs]).mean()
        loss1 = torch.stack([i["reconstruct loss"] for i in outputs]).mean()
        loss2 = torch.stack([i["predict loss"] for i in outputs]).mean()
        outputs.clear()
        # print(loss, loss1, loss2)
        return loss, loss1, loss2

    def on_training_epoch_end(self):

        loss, loss1, loss2 = self._common_epoch_end(self.training_step_outputs)

        print("finish training epoch, loss %f" % loss)
        # self.log_dict(
        #     {
        #         "epoch_train_loss": loss,
        #     },
        #     on_step=False,
        #     on_epoch=True,
        #     prog_bar=True,
        # )

        self.last_train_step = 0

    def on_validation_epoch_end(self):
        loss, loss1, loss2 = self._common_epoch_end(self.validation_step_outputs)
        print("finish validating, loss %f" % (loss))
        self.log_dict(
            {
                "epoch_validate_loss": loss1 + loss2,
                "epoch_validate_reconstruct_loss": loss1,
                "epoch_validate_predict_loss": loss2,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

    def validation_step(self, batch, batch_idx):

        input_dict, labels = batch
        y = input_dict["ori_seq_t"]
        mask = input_dict["mask"].float()

        loss, loss1, loss2 = self._common_training_step(input_dict, y, mask, labels)
        self.validation_step_outputs.append(
            {
                "total loss": loss.detach().cpu(),
                "predict loss": loss2.detach().cpu(),
                "reconstruct loss": loss1.detach().cpu(),
            }
        )

        return loss

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i in checkpoint["state_dict"]:
            if "esm" in i and "lora" not in i:
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]

    def configure_optimizers(self):

        print("get training optimizer")

        if self.lr_backbone is not None:
            l1 = []
            l2 = []
            for i, j in self.named_parameters():
                if j.requires_grad:
                    if "esm" in i:
                        l1.append(j)
                    else:
                        l2.append(j)

            param_dicts = [
                {
                    "params": l1,
                    "lr": self.lr_backbone,
                },
                {
                    "params": l2,
                    "lr": self.lr,
                },
            ]
            return torch.optim.Adam(param_dicts, weight_decay=self.weight_decay)

        if self.load_freeze is None:
            t = []
            for i, j in self.named_parameters():
                if j.requires_grad:
                    print(i)
                    t.append(j)
            optimizer = torch.optim.Adam(
                t,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        else:
            l1 = []
            for i, j in self.named_parameters():
                if "esm_model" not in i or ("output_heads" in i and "lora" in i):
                    print(i)
                    l1.append(j)
            optimizer = torch.optim.Adam(
                l1,
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
            for need in self.load_freeze:
                params = []
                for i, j in self.named_parameters():
                    flag = 1
                    for k in need:
                        if k not in i:
                            flag = 0
                            break
                    if flag == 1:
                        print(i)
                        params.append(j)
                optimizer.add_param_group({"params": params, "lr": self.lr})

        return optimizer


@dataclass
class VESMOutputs:
    S1Embeddings: dict[str, torch.Tensor]
    S1Logits: dict[str, torch.Tensor]
    S1Predicts: dict[str, dict[str, torch.Tensor]]
    S2Embeddings: torch.Tensor | None
    S2Reconstruct: dict[str, torch.Tensor] | None
    S2Logits: dict[str, torch.Tensor] | None
    S2Predicts: dict[str, torch.Tensor] | None


@dataclass
class VESMLosses:
    S1PredictsLosses: dict[str, torch.Tensor] | None
    S1LogitsLosses: dict[str, torch.Tensor] | None
    S2ReconstructLosses: dict[str, torch.Tensor] | None
    S2LogitsLosses: dict[str, torch.Tensor] | None
    S2PredictsLoss: torch.Tensor | None


@dataclass(frozen=True)
class VESMConfig:
    prots: list[str]
    ori_seqs: dict[str, torch.Tensor]

    # stage 1
    esm_model_type: str
    esm_model_channels: int
    out_channels: int = 256
    aa_counts: int = 33
    stage_1_transformer_layers: int = 3
    stage_1_clf_hidden_dim: int
    teaching_force: float = 0.5

    # stage 2
    stage_2_clf_hidden_dim: int
    n_head: int = 16
    stage_2_transformer_layers: int = 5

    # training params
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 0.0
    stage_1_masked_weight: float = 0.1
    stage_2_masekd_weight: float = 2.0
    stage_1_regressor_weight: float = 1.0
    stage_2_recosntruct_weight: float = 1.0
    stage_2_regressor_weight: float = 1.0


class Stage1Regressors(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_dim,
    ):
        super().__init__()
        self.fabricated = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.time_series = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return {"fabricated": self.fabricated(x), "time_series": self.time_series(x)}


class Stage2Regressors(nn.Module):
    def __init__(self, out_channels, hidden_dim):
        super().__init__()
        self.time_series = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return {"time_series": self.time_series(x)}


class VESM(L.LightningModule):
    def __init__(self, esm_model, stage, config: VESMConfig):
        super().__init__()

        assert stage in [
            "trainint stage 1",
            "training stage 2",
            "training stage 1 + stage 2",
            "inference",
        ]
        self.stage = stage
        self.config = config
        self.prots = config.prots

        self.esm_model = ESMModule(esm_model, config.esm_model_type)

        # stage 1 modules
        self.stage_1_bottleneck = nn.Linear(
            config.esm_model_channels, config.out_channels
        )

        self.stage_1_reconstructor = DecoderBlock(
            config.esm_model_channels,
            config.n_head,
            config.aa_counts,
            config.stage_1_transformer_layers,
        )

        self.ori_seqs = {}
        for i in config.ori_seqs:
            t = torch.tensor(config.ori_seqs[i], requires_grad=False)
            self.ori_seqs[i] = t
            self.register_buffer("ori_seq_" + i, t)

        self.stage_1_embed = nn.Embedding(config.aa_counts, config.out_channels)

        self.stage_1_regressors = Stage1Regressors(
            config.out_channels,
            config.stage_1_clf_hidden_dim,
            config.stage1_downstream_task_dim,
        )

        # stage 2 modules
        self.stage_2_encoder_blocks = nn.ModuleList(
            [
                SelfAttention(config.out_channels, config.n_head)
                for i in range(config.stage_2_transformer_layers)
            ]
        )

        self.stage_2_decoder_blocks = nn.ModuleList(
            [
                SelfAttention(config.out_channels, config.n_head)
                for i in range(config.stage_2_transformer_layers)
            ]
        )

        self.stage_2_regressors = Stage2Regressors(
            config.out_channels,
            config.stage_2_clf_hidden_dim,
            config.stage_2_downstream_task_dim,
        )

        self.stage_2_reconstructor = DecoderBlock(
            config.out_channels,
            config.n_head,
            config.aa_counts,
            config.stage_2_transformer_layers,
        )

        self.stage_2_embed = nn.Embedding(config.aa_counts, config.out_channels)

        # training utils
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.training_step_outputs = []
        self.validation_step_outputs = []

    # masks not used
    def stage1_forward(self, input_dict, masks=None):
        stage_1_embeds = {}
        stage_1_logits = {}
        stage_1_predicts = {}
        stage_1_ori_embeds = {}
        for i in input_dict:
            if i not in self.prots:
                continue
            x = self.esm_model(input_dict[i])
            embed = self.stage_1_bottleneck(x)
            batchsize, length, channels = embed.shape

            stage_1_ori_embeds[i] = embed

            embed = embed[:, 0]
            stage_1_embeds[i] = embed

            predicts = self.stage_1_regressors(embed)
            if self.stage == "infernce":
                # predicts.pop("time_series")
                predicts.pop("fabricated")
            stage_1_predicts[i] = predicts

            x = embed[:, None].repeat(1, length, 1)

            if (
                self.stage == "training stage 1"
                and self.config.teaching_force > 0
                and random.random() < self.config.teaching_force
            ):
                q = getattr(self, "ori_seq_" + i)[None, :].repeat(batchsize, 1)
                q = self.stage_1_embed(q)
                l = min(x.shape[1], q.shape[1])
                if x.shape[1] < q.shape[1]:
                    q[:, :l, :] += x[:, :l, :]
                    x = q
                else:
                    x[:, :l, :] += q[:, :l, :]
            x = self.stage_1_reconstructor(x)
            stage_1_logits[i] = x

        return stage_1_ori_embeds, stage_1_embeds, stage_1_logits, stage_1_predicts

    def stage2_forward(self, stage_1_embeds, masks=None):
        if masks is None:
            masks = []
        inputs = []
        batch_size = 1
        for i in self.prots:
            if i in stage_1_embeds:
                batch_size = stage_1_embeds[i].shape[0]
                break

        masked = torch.zeros(batch_size, self.config.out_channels).to(self.device)
        for i in self.prots:
            if i in stage_1_embeds and i not in masks:
                inputs.append(stage_1_embeds[i])
            else:
                inputs.append(masked)
        inputs.append(masked)
        inputs = torch.stack(inputs, dim=1)

        for block in self.stage_2_encoder_blocks:
            inputs = block(inputs)

        embed = inputs[:, -1, :]

        stage_2_embeddings = embed

        stage_2_logits = {}

        for i in self.prots:
            q = getattr(self, "ori_seq_" + i)[None, :].repeat(batch_size, 1)
            q = self.stage_2_embed(q)
            length = q.shape[1]
            if i in stage_1_embeds:
                length = stage_1_embeds[i].shape[1]
            x = embed[:, None].repeat(1, length, 1)
            l = min(x.shape[1], q.shape[1])
            if x.shape[1] < q.shape[1]:
                q[:, :l, :] += x[:, :l, :]
                x = q
            else:
                x[:, :l, :] += q[:, :l, :]
            x = self.stage_2_reconstructor(x)
            stage_2_logits[i] = x

        stage_2_reconstruct = {}

        embeded = embed[:, None].repeat(1, len(self.prots), 1)
        for block in self.stage_2_decoder_blocks:
            embeded = block(embeded)

        for i, s in zip(range(len(self.prots)), self.prots):
            stage_2_reconstruct[s] = embeded[:, i]

        stage_2_predicts = self.stage_2_regressors(embed)

        return stage_2_embeddings, stage_2_reconstruct, stage_2_logits, stage_2_predicts

    def forward(self, input_dict, stage_1_masks=None, stage_2_masks=None):
        if "stage 1" in self.stage:
            stage_1_ori_embeds, stage_1_embeds, stage_1_logits, stage_1_predicts = (
                self.stage1_forward(input_dict, stage_1_masks)
            )
            if self.stage == "training stage 1":
                return VESMOutputs(
                    S1Embeddings=stage_1_embeds,
                    S1Logits=stage_1_logits,
                    S1Predicts=stage_1_predicts,
                    S2Embeddings=None,
                    S2Reconstruct=None,
                    S2Logits=None,
                    S2Predicts=None,
                )
        else:
            with torch.no_grad():
                stage_1_ori_embeds, stage_1_embeds, stage_1_logits, stage_1_predicts = (
                    self.stage1_forward(input_dict, stage_1_masks)
                )

        if "stage 2" in self.stage:

            (
                stage_2_embeddings,
                stage_2_reconstruct,
                stage_2_logits,
                stage_2_predicts,
            ) = self.stage2_forward(stage_1_ori_embeds, stage_2_masks)
        else:
            with torch.no_grad():
                (
                    stage_2_embeddings,
                    stage_2_reconstruct,
                    stage_2_logits,
                    stage_2_predicts,
                ) = self.stage2_forward(stage_1_ori_embeds, stage_2_masks)

        return VESMOutputs(
            S1Embeddings=stage_1_embeds,
            S1Logits=stage_1_logits,
            S1Predicts=stage_1_predicts,
            S2Embeddings=stage_2_embeddings,
            S2Reconstruct=stage_2_reconstruct,
            S2Logits=stage_2_logits,
            S2Predicts=stage_2_predicts,
        )

    def stage1_time_series_loss(self, output1, output2):
        stage_1_time_series_loss = {}
        for i in output1.S1Predicts:
            t1 = output1.S1Predicts[i]["time_series"]
            t2 = output2.S1Predicts[i]["time_series"]
            t = t2 - t1
            stage_1_time_series_loss[i] = self.bce(t, torch.ones_like(t))

        return stage_1_time_series_loss

    def stage1_frabricated_loss(self, output, input_dict):
        stage_1_frabricated_loss = {}
        for i in output.S1Predicts:
            t = input_dict["labels"][i]
            if t.dim() == 1:
                t = t.unsqueeze(0)
            loss = self.bce(
                output.S1Predicts[i]["fabricated"],
                input_dict["labels"][i],
            )
            stage_1_frabricated_loss[i] = loss
        return stage_1_frabricated_loss

    def stage1_logit_loss(self, output: VESMOutputs, input_dict):
        stage_1_logitLosses = {}
        if "ori_inputs" not in input_dict:
            return stage_1_logitLosses
        for i in output.S1Logits:
            loss = self.cross_entropy(
                output.S1Logits[i].flatten(),
                input_dict["ori_inputs"][i].flatten(),
            )
            if input_dict["stage_1_masks"] is not None:
                mask = input_dict["stage_1_masks"][i].flatten()
                mask = 1 - mask
                mask[mask < 0] = -self.config.stage_1_masked_weight
                mask += self.config.stage_1_masked_weight
            else:
                mask = torch.ones_like(loss)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
            stage_1_logitLosses[i] = loss
        return stage_1_logitLosses

    def stage2_logit_loss(self, output: VESMOutputs, input_dict):
        stage_2_logitLosses = {}
        if "ori_inputs" not in input_dict:
            return stage_2_logitLosses
        for i in output.S2Logits:
            loss = self.cross_entropy(
                output.S2Logits[i].flatten(),
                input_dict["ori_inputs"][i].flatten(),
            )
            if input_dict["stage_1_masks"] is not None:
                mask = input_dict["stage_1_masks"][i].flatten()
                mask = 1 - mask
                mask[mask < 0] = -self.config.stage_1_masked_weight
                mask += self.config.stage_1_masked_weight
            else:
                mask = torch.ones_like(loss)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
            stage_2_logitLosses[i] = loss
        return stage_2_logitLosses

    def stage2_time_series_loss(self, output1, output2):
        t1 = output1.S2Predicts["time_series"]
        t2 = output2.S2Predicts["time_series"]
        t = t2 - t1
        return self.bce(t, torch.ones_like(t))

    def stage2_reconstruct_loss(self, output, input_dict):
        stage_2_reconstruct_loss = {}
        for i in output.S2Reconstruct:
            loss = self.mse(
                output.S2Reconstruct[i].flatten(),
                output.S1Embeddings[i].flatten(),
            )
            if i in input_dict["stage_2_masks"]:
                loss *= self.config.stage_2_masekd_weight
            stage_2_reconstruct_loss[i] = loss

    def getLoss(self, input_dict1, input_dict2=None):
        output1 = self.forward(
            input_dict1["inputs"],
            input_dict1["stage_1_masks"],
            input_dict1["stage_2_masks"],
        )
        if input_dict2 is not None:
            output2 = self.forward(
                input_dict2["inputs"],
                input_dict2["stage_1_masks"],
                input_dict2["stage_2_masks"],
            )
        else:
            output2 = None

        stage_1_logitLosses = {}
        stage_1_predictLosses = {}
        s = self.stage1_logit_loss(output1, input_dict1)
        for i in s:
            stage_1_logitLosses[i + "_1"] = s[i]
        if output2 is not None:
            s = self.stage1_logit_loss(output2, input_dict2)
            for i in s:
                stage_1_logitLosses[i + "_2"] = s[i]

        stage_1_frabricated_loss = self.stage1_frabricated_loss(output1, input_dict1)
        for i in stage_1_frabricated_loss:
            stage_1_predictLosses[i + "_fabricated_1"] = stage_1_frabricated_loss[i]

        if output2 is not None:
            stage_1_frabricated_loss = self.stage1_frabricated_loss(
                output2, input_dict2
            )
            for i in stage_1_frabricated_loss:
                stage_1_predictLosses[i + "_fabricated_2"] = stage_1_frabricated_loss[i]

        if output2 is not None:
            s = self.stage1_time_series_loss(output1, output2)
            for i in s:
                stage_1_predictLosses[i + "_time_series"] = s[i]

        if self.stage == "training stage 1":
            return VESMLosses(
                S1PredictsLosses=stage_1_predictLosses,
                S1LogitsLosses=stage_1_logitLosses,
                S2ReconstructLosses=None,
                S2LogitsLosses=None,
                S2PredictsLoss=None,
            )

        stage_2_reconstruct_loss = {}
        s = self.stage2_reconstruct_loss(output1, input_dict1)
        for i in s:
            stage_2_reconstruct_loss[i + "_1"] = s[i]
        if output2 is not None:
            s = self.stage2_reconstruct_loss(output2, input_dict2)
            for i in s:
                stage_2_reconstruct_loss[i + "_2"] = s[i]

        stage_2_logitLosses = {}
        s = self.stage2_logit_loss(output1, input_dict1)
        for i in s:
            stage_2_logitLosses[i + "_1"] = s[i]
        if output2 is not None:
            s = self.stage2_logit_loss(output2, input_dict2)
            for i in s:
                stage_2_logitLosses[i + "_2"] = s[i]
        if output2 is not None:
            stage_2_predictLosses = self.stage2_time_series_loss(output1, output2)
        else:
            stage_2_predictLosses = None

        return VESMLosses(
            S1PredictsLosses=stage_1_predictLosses,
            S1LogitsLosses=stage_1_logitLosses,
            S2ReconstructLosses=stage_2_reconstruct_loss,
            S2LogitsLosses=stage_2_logitLosses,
            S2PredictsLoss=stage_2_predictLosses,
        )

    def _common_training_step(self, input_dict1, input_dict2=None):
        loss = self.getLoss(input_dict1, input_dict2)

        if self.stage == "training stage 1":
            loss1 = sum([i for i in loss.S1PredictsLosses.values()])
            loss2 = sum([i for i in loss.S1LogitsLosses.values()])
            loss = loss1 * self.config.stage_1_regressor_weight + loss2
            d = {
                "S1PredictsLosses": loss1.detach().cpu(),
                "S1LogitsLosses": loss2.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
            return loss, d

        if self.stage == "training stage 2":
            loss1 = sum([i for i in loss.S2ReconstructLosses.values()])
            loss2 = sum([i for i in loss.S2LogitsLosses.values()])
            loss3 = loss.S2PredictsLoss
            loss = (
                loss1 * self.config.stage_2_recosntruct_weight
                + loss2
                + loss3 * self.config.stage_2_regressor_weight
            )
            d = {
                "S2ReconstructLosses": loss1.detach().cpu(),
                "S2LogitsLosses": loss2.detach().cpu(),
                "S2PredictsLoss": loss3.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
            return loss, d

        if self.stage == "training stage 1 + stage 2":
            loss1 = sum([i for i in loss.S1PredictsLosses.values()])
            loss2 = sum([i for i in loss.S1LogitsLosses.values()])
            loss3 = sum([i for i in loss.S2ReconstructLosses.values()])
            loss4 = sum([i for i in loss.S2LogitsLosses.values()])
            loss5 = loss.S2PredictsLoss
            loss = (
                loss1 * self.config.stage_1_regressor_weight
                + loss2
                + loss3 * self.config.stage_2_recosntruct_weight
                + loss4
                + loss5 * self.config.stage_2_regressor_weight
            )
            d = {
                "S1PredictsLosses": loss1.detach().cpu(),
                "S1LogitsLosses": loss2.detach().cpu(),
                "S2ReconstructLosses": loss3.detach().cpu(),
                "S2LogitsLosses": loss4.detach().cpu(),
                "S2PredictsLoss": loss5.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
            return loss, d
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            input_dict1, input_dict2 = batch
        else:
            input_dict1 = batch
            input_dict2 = None
        loss, d = self._common_training_step(input_dict1, input_dict2)
        dp = {}
        for i in d:
            dp["training_" + i] = d[i]
        self.training_step_outputs.append(dp)
        return loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, tuple):
            input_dict1, input_dict2 = batch
        else:
            input_dict1 = batch
            input_dict2 = None
        loss, d = self._common_training_step(input_dict1, input_dict2)
        dp = {}
        for i in d:
            dp["validation_" + i] = d[i]
        self.validation_step_outputs.append(dp)
        return loss

    def _common_epoch_end(self, outputs):
        if len(outputs) == 0:
            return {}
        res = {}
        for i in outputs:
            for j in i:
                if j not in res:
                    res[j] = []
                res[j].append(i[j])

        for i in res:
            res[i] = torch.stack(res[i]).mean()
        outputs.clear()
        return res

    def on_training_epoch_end(self):
        res = self._common_epoch_end(self.training_step_outputs)
        print("finish traing epoch with loss:")
        print(res)
        for i in res:
            self.log(i, res[i], prog_bar=False)
        self.last_train_step = 0

    def on_validation_epoch_end(self):
        res = self._common_epoch_end(self.validation_step_outputs)
        print("finish validating epoch with loss:")
        print(res)
        for i in res:
            self.log(i, res[i], prog_bar=False)

    def on_before_optimizer_step(self, optimizer) -> None:
        res = {}
        for i in self.training_step_outputs[self.last_train_step :]:
            for j in i:
                if j not in res:
                    res[j] = []
                res[j].append(i[j])
        for i in res:
            res[i] = torch.stack(res[i]).mean()

        for i in res:
            self.log(i, res[i], prog_bar=True)

    def configure_optimizers(self):
        if self.config.lr_backbone is not None:
            l1 = []
            l2 = []
            for i, j in self.named_parameters():
                if j.requires_grad:
                    if "esm" in i:
                        l1.append(j)
                    else:
                        l2.append(j)

            param_dicts = [
                {
                    "params": l1,
                    "lr": self.config.lr_backbone,
                },
                {
                    "params": l2,
                    "lr": self.config.lr,
                },
            ]
            return torch.optim.Adam(param_dicts, weight_decay=self.config.weight_decay)

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i in checkpoint["state_dict"]:
            if "esm" in i and "lora" not in i:
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]


class ESMModule(nn.Module):
    def __init__(self, esm_model, esm_model_type):
        super().__init__()
        self.esm_model = esm_model
        self.esm_model_type = esm_model_type

    def forward(self, input_dict):
        if self.esm_model_type == "esm3":
            for i in ["seq_t", "structure_t", "ss8_t", "sasa_t"]:
                if i not in input_dict:
                    input_dict[i] = None
                else:
                    if len(input_dict[i].size()) == 1:
                        input_dict[i] = input_dict[i].unsqueeze(0)

            representations = self.esm_model(
                sequence_tokens=input_dict["seq_t"],
                structure_tokens=input_dict["structure_t"],
                ss8_tokens=input_dict["ss8_t"],
                sasa_tokens=input_dict["sasa_t"],
            )
            x = representations.embeddings
            return x

        if self.esm_model_type == "esm2":
            assert "seq_t" in input_dict
            t = input_dict["seq_t"]
            representations = self.esm_model(t, repr_layers=[self.num_layers])

            x = representations["representations"][self.num_layers][:, 0]
            return x

        if self.esm_model_type == "esmc":
            assert "seq_t" in input_dict
            t = input_dict["seq_t"]
            if len(t.size()) == 1:
                t = t.unsqueeze(0)

            representations = self.esm_model(
                sequence_tokens=t,
            )

            x = representations.embeddings
            return x

        raise NotImplementedError


class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim) * std_dev)
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)


def _set_submodule(submodule, module_path, new_module):
    tokens = module_path.split(".")
    for token in tokens[:-1]:
        submodule = getattr(submodule, token)
    setattr(submodule, tokens[-1], new_module)


def addlora(esm_model, layers, ranks, alphas):
    # if layers is None:
    #     layers = [str(i) for i in range(12)]
    for i, j in esm_model.named_modules():
        if isinstance(j, nn.Linear):
            # print(i)
            # res = [False]
            # res.extend([t in i for t in layers])
            # res = reduce(lambda x, y: x or y, res)
            for layer, rank, alpha in zip(layers, ranks, alphas):
                if str(layer) in i:
                    _set_submodule(
                        esm_model,
                        i,
                        LinearWithLoRA(j, rank, alpha),
                    )
    return esm_model
