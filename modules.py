import random
from types import FunctionType as function

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchtune
from attr import dataclass


@dataclass
class VESMOutputs:
    S1Embeddings: dict[str, torch.Tensor]
    S1Logits: dict[str, torch.Tensor]
    S1Predicts_aa: dict[str, torch.Tensor]
    S1Predicts: dict[str, dict[str, torch.Tensor]]
    S2Embeddings: torch.Tensor | None
    S2Reconstruct: dict[str, torch.Tensor] | None
    S2Predicts: dict[str, torch.Tensor] | None


@dataclass
class VESMLosses:
    S1PredictsLosses: dict[str, torch.Tensor] | None
    S1PredictsAALosses: dict[str, torch.Tensor] | None
    S1LogitsLosses: dict[str, torch.Tensor] | None
    S2ReconstructLosses: dict[str, torch.Tensor] | None
    S2PredictsLoss: torch.Tensor | None


@dataclass(frozen=True)
class VESMConfig:
    prots: list[str]

    # stage 1
    esm_model_type: str
    esm_model_channels: int
    out_channels: int = 512

    # aa wise predicts
    aa_counts: int = 33
    aa_predict_classes: int = 3

    # protein wise predicts
    stage1_predict_classes: int = 0

    stage_1_transformer_layers: int = 5
    stage_1_clf_hidden_dim: int = 512
    teaching_force: float = 0.5

    # stage 2
    stage_2_clf_hidden_dim: int = 512
    n_head: int = 16
    stage_2_transformer_layers: int = 5
    stage2_predict_classes: int = 0

    # training params
    lr: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 0.0
    stage_1_masked_weight: float = 0.1
    stage_2_masked_weight: float = 2.0
    stage_1_regressor_weight: float = 1.0
    stage_2_recosntruct_weight: float = 1.0
    stage_2_regressor_weight: float = 1.0
    stage_1_predict_loss: function | None = None
    stage_2_predict_loss: function | None = None


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
    def __init__(
        self, channels, n_head, aa_classes, predict_classes, transformer_layers=3
    ):
        super(DecoderBlock, self).__init__()
        self.channels = channels
        self.n_head = n_head
        self.aa_classes = aa_classes
        self.predict_classes = predict_classes

        self.transformer_blocks = nn.ModuleList(
            [SelfAttention(channels, n_head) for i in range(transformer_layers)]
        )
        # self.T1 = SelfAttention(channels, n_head)
        # self.T2 = SelfAttention(channels, n_head)
        # self.T3 = SelfAttention(channels, n_head)
        self.aa_clf = nn.Linear(channels, aa_classes)
        self.clf = nn.Linear(channels, predict_classes)

    def forward(self, x):
        for block in self.transformer_blocks:
            x = block(x)
        aa_1 = self.clf(x)
        aa_2 = self.aa_clf(x)
        return {"predict_logits": aa_1, "aa_logits": aa_2}  # .swapaxes(2, 1)


class Linearlayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2, layer_norm=False, activate="gelu"):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if layer_norm is not None:
            self.ln = nn.LayerNorm(out_dim)
        else:
            self.ln = None
        if dropout > 0 and dropout < 1:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        if activate == "gelu":
            self.activate = nn.GELU()
        elif activate == "relu":
            self.activate = nn.ReLU()
        elif activate == "leakyrelu":
            self.activate = nn.LeakyReLU()
        else:
            self.activate = nn.Identity()
            # raise ValueError("activate %s not supported" % acivate)
        # self.activate = activate

    def forward(self, x):
        x = self.linear(x)
        if self.ln is not None:
            x = self.ln(x)
        x = self.activate(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class Linearcls(nn.Module):
    """simple linear classifier

    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim=256,
        take_embed="first",
        dropout=-1,
        p0=None,
        output_dim=1,
        hidden_dim=256,
        hidden_layer=-1,
        activate="gelu",
        layer_norm=True,
    ):
        super().__init__()

        assert take_embed in ["first", "mean", "max"]
        self.embed_dim = input_dim
        self.dropout = dropout
        self.take_embed = take_embed
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if hidden_layer == -1:
            self.l1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
            self.l2 = nn.Linear(self.embed_dim // 2, self.embed_dim // 4)
            self.l3 = nn.Linear(self.embed_dim // 4, output_dim)
            self.ln1 = nn.LayerNorm(self.embed_dim // 2)
            self.ln2 = nn.LayerNorm(self.embed_dim // 4)

            if dropout > 0 and dropout < 1:
                self.dropout1 = nn.Dropout(p=self.dropout)
                self.dropout2 = nn.Dropout(p=self.dropout)
            else:
                self.dropout1 = None
                self.dropout2 = None
            self.layers = None
        else:

            in_dims = [input_dim] + [hidden_dim] * (hidden_layer)
            output_dims = [hidden_dim] * (hidden_layer + 1)
            self.layers = nn.ModuleList(
                [
                    Linearlayer(
                        in_dims[i],
                        output_dims[i],
                        dropout=dropout,
                        layer_norm=layer_norm,
                        activate=activate,
                    )
                    for i in range(len(in_dims))
                ]
            )
            self.output = nn.Linear(hidden_dim, output_dim)

        if p0 is None:
            self.p0 = None
        else:
            self.p0 = nn.Dropout(p0)

    def forward(self, x: torch.Tensor):

        if self.take_embed == "first":
            x = x[:, 0]
        elif self.take_embed == "mean":
            x = torch.mean(x, dim=1)
        elif self.take_embed == "max":
            x = x.transpose(1, 2)
            x = F.adaptive_max_pool1d(x, 1)

        if self.p0 is not None:
            x = self.p0(x)

        if self.layers is None:
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
            return x

        for layer in self.layers:
            x = layer(x)
        # print("lin", x.shape)
        x = self.output(x)
        return x
        if self.output_dim == 1:
            return x
        else:
            return x[:, 0], x[:, 1:]


class Regressors(nn.Module):
    def __init__(self, out_channels, hidden_dim, predict_classes):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, predict_classes),
        )
        self.time_series = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return {"predictions": self.clf(x), "time_series": self.time_series(x)}


class VESM(L.LightningModule):
    def __init__(self, esm_model, stage, config: VESMConfig):
        super().__init__()

        assert stage in [
            "pretraining stage 1",
            "training stage 1",
            "training stage 2",
            "training stage 1 + stage 2",
            "inference",
        ]
        self.stage = stage

        print("model at stage:", stage)

        self.config = config
        self.prots = config.prots

        self.esm_model = ESMModule(esm_model, config.esm_model_type)

        # stage 1 modules
        self.stage_1_bottleneck = nn.Linear(
            config.esm_model_channels, config.out_channels
        )

        self.stage_1_reconstructor = DecoderBlock(
            config.out_channels,
            config.n_head,
            config.aa_counts,
            config.aa_predict_classes,
            config.stage_1_transformer_layers,
        )

        self.stage_1_embed = nn.Embedding(config.aa_counts, config.out_channels)

        self.stage_1_regressors = Regressors(
            config.out_channels,
            config.stage_1_clf_hidden_dim,
            config.stage1_predict_classes,
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

        self.stage_2_regressors = Regressors(
            config.out_channels,
            config.stage_2_clf_hidden_dim,
            config.stage2_predict_classes,
        )

        # training utils
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.cross_entropy_mutation = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1, 1.0, 1.0])
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.last_train_step = 0

    def stage1_forward(self, input_dict, masks=None):
        stage_1_embeds = {}
        stage_1_logits = {}
        stage_1_predicts = {}
        stage_1_aa_clf = {}
        for i in input_dict:
            if i not in self.prots:
                continue
            # print(input_dict[i])
            x = self.esm_model(input_dict[i])
            embed = self.stage_1_bottleneck(x)
            batchsize, length, channels = embed.shape

            embed = embed[:, 0]
            stage_1_embeds[i] = embed

            predicts = self.stage_1_regressors(embed)
            if self.stage == "inference":
                # predicts.pop("time_series")
                predicts.pop("fabricated")
            stage_1_predicts[i] = predicts

            x = embed[:, None].repeat(1, length, 1)

            if (
                self.config.teaching_force > 0.0
                and random.random() < self.config.teaching_force
            ):
                if "aligned_" + i in input_dict:
                    q = input_dict["aligned_" + i]
                    if q.dim() == 1:
                        q = q[None, :].repeat(batchsize, 1)

                    q = self.stage_1_embed(q)
                    x += q
            # print(x.shape, stage_1_ori_embeds[i].shape)
            x = self.stage_1_reconstructor(x)
            stage_1_logits[i] = x["aa_logits"]
            stage_1_aa_clf[i] = x["predict_logits"]

        return stage_1_embeds, stage_1_logits, stage_1_predicts, stage_1_aa_clf

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

        # placeholder for global embedding
        inputs.append(masked)

        for i in self.prots:
            if i in stage_1_embeds and i not in masks:
                inputs.append(stage_1_embeds[i])
            else:
                inputs.append(masked)

        inputs.append(masked)
        inputs = torch.stack(inputs, dim=1)

        for block in self.stage_2_encoder_blocks:
            inputs = block(inputs)

        embed = inputs[:, 0, :]

        stage_2_embeddings = embed

        stage_2_reconstruct = {}

        embeded = embed[:, None].repeat(1, len(self.prots), 1)
        for block in self.stage_2_decoder_blocks:
            embeded = block(embeded)

        for i, s in zip(range(len(self.prots)), self.prots):
            stage_2_reconstruct[s] = embeded[:, i]

        stage_2_predicts = self.stage_2_regressors(embed)

        return stage_2_embeddings, stage_2_reconstruct, stage_2_predicts

    def forward(
        self, input_dict, stage_1_masks=None, stage_2_masks=None, only_stage_1=False
    ):
        if "stage 1" in self.stage:
            stage_1_embeds, stage_1_logits, stage_1_predicts, stage_1_aa_clf = (
                self.stage1_forward(input_dict, stage_1_masks)
            )
            if self.stage == "training stage 1" or self.stage == "pretraining stage 1":
                return VESMOutputs(
                    S1Embeddings=stage_1_embeds,
                    S1Logits=stage_1_logits,
                    S1Predicts=stage_1_predicts,
                    S1Predicts_aa=stage_1_aa_clf,
                    S2Embeddings=None,
                    S2Reconstruct=None,
                    S2Predicts=None,
                )
        else:
            with torch.no_grad():
                stage_1_embeds, stage_1_logits, stage_1_predicts, stage_1_aa_clf = (
                    self.stage1_forward(input_dict, stage_1_masks)
                )

        if only_stage_1:
            return VESMOutputs(
                S1Embeddings=stage_1_embeds,
                S1Logits=stage_1_logits,
                S1Predicts=stage_1_predicts,
                S1Predicts_aa=stage_1_aa_clf,
                S2Embeddings=None,
                S2Reconstruct=None,
                S2Predicts=None,
            )

        if "stage 2" in self.stage:

            (
                stage_2_embeddings,
                stage_2_reconstruct,
                stage_2_predicts,
            ) = self.stage2_forward(stage_1_embeds, stage_2_masks)
        else:
            with torch.no_grad():
                (
                    stage_2_embeddings,
                    stage_2_reconstruct,
                    stage_2_predicts,
                ) = self.stage2_forward(stage_1_embeds, stage_2_masks)

        return VESMOutputs(
            S1Embeddings=stage_1_embeds,
            S1Logits=stage_1_logits,
            S1Predicts=stage_1_predicts,
            S1Predicts_aa=stage_1_aa_clf,
            S2Embeddings=stage_2_embeddings,
            S2Reconstruct=stage_2_reconstruct,
            S2Predicts=stage_2_predicts,
        )

    def stage1_time_series_loss(self, output1: VESMOutputs, output2: VESMOutputs):
        stage_1_time_series_loss = {}
        for i in output1.S1Predicts:
            t1 = output1.S1Predicts[i]["time_series"]
            t2 = output2.S1Predicts[i]["time_series"]
            t = t2 - t1
            stage_1_time_series_loss[i] = self.bce(t, torch.ones_like(t))

        return stage_1_time_series_loss

    def stage1_prediction_loss(self, output: VESMOutputs, input_dict):
        stage_1_prediction_loss = {}
        if "label" not in input_dict or self.config.stage1_predict_classes == 0:
            return stage_1_prediction_loss
        for i in output.S1Predicts:
            t = input_dict["label"][i][0].float()
            if t.dim() == 1:
                t = t.unsqueeze(0)

            if self.config.stage_1_predict_loss is not None:
                loss = self.config.stage_1_predict_loss(
                    output.S1Predicts[i]["predictions"], t
                )
            else:
                loss = self.bce(
                    output.S1Predicts[i]["predictions"],
                    t,
                )
            stage_1_prediction_loss[i] = loss
        return stage_1_prediction_loss

    def stage1_logit_loss(self, output: VESMOutputs, input_dict):
        stage_1_logitLosses = {}
        if "ori_seq" not in input_dict:
            return stage_1_logitLosses
        for i in output.S1Logits:
            s1 = output.S1Logits[i]
            s1 = s1.view(-1, s1.shape[-1])
            l1 = input_dict["ori_seq"][i].flatten()
            loss = self.cross_entropy(s1, l1)
            if (
                "stage_1_masks" in input_dict
                and input_dict["stage_1_masks"] is not None
            ):
                mask = input_dict["stage_1_masks"][i].flatten().float()
                mask = 1 - mask
                mask[mask < 0] = -self.config.stage_1_masked_weight
                mask += self.config.stage_1_masked_weight
            else:
                mask = torch.ones_like(loss)
            loss = loss * mask
            loss = loss.sum() / mask.sum()
            stage_1_logitLosses[i] = loss
        return stage_1_logitLosses

    def stage1_aa_prediction_loss(self, output: VESMOutputs, input_dict):
        stage_1_aa_logitLosses = {}
        if "mutation_label" not in input_dict:
            return stage_1_aa_logitLosses
        for i in output.S1Predicts_aa:
            s1 = output.S1Predicts_aa[i]
            s1 = s1.view(-1, s1.shape[-1])
            l1 = input_dict["mutation_label"][i].flatten()
            # print(s1, l1)
            # print(s1.shape, l1.shape)
            loss = self.cross_entropy_mutation(s1, l1)
            stage_1_aa_logitLosses[i] = loss
        return stage_1_aa_logitLosses

    def stage2_time_series_loss(self, output1: VESMOutputs, output2: VESMOutputs):
        t1 = output1.S2Predicts["time_series"]
        t2 = output2.S2Predicts["time_series"]
        t = t2 - t1
        return self.bce(t, torch.ones_like(t))

    def stage2_reconstruct_loss(self, output: VESMOutputs, input_dict):
        stage_2_reconstruct_loss = {}
        for i in output.S2Reconstruct:
            loss = self.mse(
                output.S2Reconstruct[i].flatten(),
                output.S1Embeddings[i].flatten(),
            )
            if i in input_dict["stage_2_masks"]:
                loss *= self.config.stage_2_masekd_weight
            stage_2_reconstruct_loss[i] = loss
        return stage_2_reconstruct_loss

    def stage2_prediction_loss(self, output: VESMOutputs, input_dict):
        stage_2_prediction_loss = None
        if "label" not in input_dict:
            return stage_2_prediction_loss
        t = input_dict["label"]["stage2"].float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if self.config.stage_2_predict_loss is not None:
            loss = self.config.stage_2_predict_loss(output.S2Predicts["predictions"], t)
        else:
            loss = self.bce(
                output.S2Predicts["predictions"],
                t,
            )
        return loss

    def getLoss(self, input_dict1, input_dict2=None):

        output1 = self.forward(
            input_dict1["input"],
            input_dict1.get("stage_1_masks", None),
            input_dict1.get("stage_2_masks", None),
        )
        if input_dict2 is not None:
            output2 = self.forward(
                input_dict2["input"],
                input_dict2.get("stage_1_masks", None),
                input_dict2.get("stage_2_masks", None),
            )
        else:
            output2 = None

        # stage 1 losses
        stage_1_logitLosses = {}
        stage_1_predictLosses = {}
        stage_1_predictAALosses = {}

        s = self.stage1_aa_prediction_loss(output1, input_dict1)
        for i in s:
            stage_1_predictAALosses[i + "_1"] = s[i]
        if output2 is not None:
            s = self.stage1_aa_prediction_loss(output2, input_dict2)
            for i in s:
                stage_1_predictAALosses[i + "_2"] = s[i]

        s = self.stage1_logit_loss(output1, input_dict1)
        for i in s:
            stage_1_logitLosses[i + "_1"] = s[i]
        if output2 is not None:
            s = self.stage1_logit_loss(output2, input_dict2)
            for i in s:
                stage_1_logitLosses[i + "_2"] = s[i]

        if "label" in input_dict1:
            stage_1_predict_loss = self.stage1_prediction_loss(output1, input_dict1)
        else:
            stage_1_predict_loss = {}

        for i in stage_1_predict_loss:
            stage_1_predictLosses[i + "_predicted_1"] = stage_1_predict_loss[i]

        if output2 is not None:
            if "label" in input_dict2:
                stage_1_predict_loss = self.stage1_prediction_loss(output2, input_dict2)
            else:
                stage_1_predict_loss = {}
            for i in stage_1_predict_loss:
                stage_1_predictLosses[i + "_predicted_2"] = stage_1_predict_loss[i]

        if output2 is not None:
            s = self.stage1_time_series_loss(output1, output2)
            for i in s:
                stage_1_predictLosses[i + "_time_series"] = s[i]

        if self.stage == "training stage 1" or self.stage == "pretraining stage 1":
            return VESMLosses(
                S1PredictsLosses=stage_1_predictLosses,
                S1LogitsLosses=stage_1_logitLosses,
                S1PredictsAALosses=stage_1_predictAALosses,
                S2ReconstructLosses=None,
                S2PredictsLoss=None,
            )

        # if self.stage == "training stage 1 + stage 2" and "ori_seq" in input_dict1:
        #     with torch.no_grad():
        #         ori_output1 = self.forward(
        #             input_dict1["ori_seq"],
        #             input_dict1.get("stage_1_masks", None),
        #             only_stage_1=True,
        #         )
        #         output1.S1Embeddings = ori_output1.S1Embeddings

        #     if input_dict2 is not None:
        #         with torch.no_grad():
        #             ori_output2 = self.forward(
        #                 input_dict2["ori_seq"],
        #                 input_dict2.get("stage_1_masks", None),
        #                 only_stage_1=True,
        #             )
        #             output2.S1Embeddings = ori_output2.S1Embeddings

        # stage 2 losses
        stage_2_reconstruct_loss = {}
        stage_2_predictLosses = {}
        s = self.stage2_reconstruct_loss(output1, input_dict1)
        for i in s:
            stage_2_reconstruct_loss[i + "_1"] = s[i]
        if output2 is not None:
            s = self.stage2_reconstruct_loss(output2, input_dict2)
            for i in s:
                stage_2_reconstruct_loss[i + "_2"] = s[i]

        if output2 is not None:
            stage_2_predictLosses["time"] = self.stage2_time_series_loss(
                output1, output2
            )
        else:
            stage_2_predictLosses = None

        return VESMLosses(
            S1PredictsLosses=stage_1_predictLosses,
            S1LogitsLosses=stage_1_logitLosses,
            S1PredictsAALosses=stage_1_predictAALosses,
            S2ReconstructLosses=stage_2_reconstruct_loss,
            S2PredictsLoss=stage_2_predictLosses,
        )

    def _common_training_step(self, input_dict1, input_dict2=None):
        loss = self.getLoss(input_dict1, input_dict2)

        if self.stage == "training stage 1" or self.stage == "pretraining stage 1":
            if len(loss.S1PredictsLosses) == 0:
                loss1 = torch.tensor((0.0))
            else:
                loss1 = sum([i for i in loss.S1PredictsLosses.values()])
            loss2 = sum([i for i in loss.S1LogitsLosses.values()])
            loss3 = sum([i for i in loss.S1PredictsAALosses.values()])
            loss = loss1 * self.config.stage_1_regressor_weight + loss2 + loss3
            d = {
                "S1PredictsLosses": loss1.detach().cpu(),
                "S1LogitsLosses": loss2.detach().cpu(),
                "S1PredictsAALosses": loss3.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
            return loss, d

        if self.stage == "training stage 2":
            loss1 = sum([i for i in loss.S2ReconstructLosses.values()])
            loss3 = loss.S2PredictsLoss
            loss = (
                loss1 * self.config.stage_2_recosntruct_weight
                + loss3 * self.config.stage_2_regressor_weight
            )
            d = {
                "S2ReconstructLosses": loss1.detach().cpu(),
                "S2PredictsLoss": loss3.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
            return loss, d

        if self.stage == "training stage 1 + stage 2":
            loss1 = sum([i for i in loss.S1PredictsLosses.values()])
            loss2 = sum([i for i in loss.S1LogitsLosses.values()])
            loss4 = sum([i for i in loss.S1PredictsAALosses.values()])

            loss3 = sum([i for i in loss.S2ReconstructLosses.values()])

            loss5 = loss.S2PredictsLoss
            loss = (
                loss1 * self.config.stage_1_regressor_weight
                + loss2
                + loss4
                + loss3 * self.config.stage_2_recosntruct_weight
                + loss5 * self.config.stage_2_regressor_weight
            )
            d = {
                "S1PredictsLosses": loss1.detach().cpu(),
                "S1LogitsLosses": loss2.detach().cpu(),
                "S2ReconstructLosses": loss3.detach().cpu(),
                "S2PredictsLoss": loss5.detach().cpu(),
                "loss": loss.detach().cpu(),
            }
            return loss, d
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list):
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
        if isinstance(batch, list):
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

        self.last_train_step = len(self.training_step_outputs)

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

    def predict_step(self, input_dict):
        if "input" in input_dict:
            output1 = self.forward(
                input_dict["input"],
                input_dict.get("stage_1_masks", None),
                input_dict.get("stage_2_masks", None),
            )
            return output1

        else:
            output1 = self.forward(input_dict)
            return output1


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
            # assert "seq_t" in input_dict
            # t = input_dict["seq_t"]
            representations = self.esm_model(input_dict, repr_layers=[self.num_layers])

            x = representations["representations"][self.num_layers][:, 0]
            return x

        if self.esm_model_type == "esmc":
            # assert "seq_t" in input_dict
            # print(input_dict)
            t = input_dict
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
