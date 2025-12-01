import random
from types import FunctionType as function

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
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
    track: list[str] = None

    # aa wise predicts
    aa_counts: int = 33
    aa_predict_classes: int = 3

    # xiugai
    regressor_version: str = "legacy"  # 默认为"legacy"（旧版）
    dropout_rate: float = 0.2

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
    stage_2_predictLosses: function | None = None


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
        import torchtune

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
    def __init__(self, out_channels, hidden_dim, predict_classes, dropout=True):
        super().__init__()
        if dropout:
            self.dropout = nn.Dropout(p=0.2)
        else:
            self.dropout = None

        # self.clf = nn.Sequential(
        #     nn.Linear(out_channels, hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, predict_classes),
        # )
        # self.time_series = nn.Sequential(
        #     nn.Linear(out_channels, hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(hidden_dim),
        #     nn.Linear(hidden_dim, 1),
        # )

        self.clf = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, predict_classes),
        )
        # CovidFit
        # self.clf = nn.Sequential(
        #     nn.Linear(out_channels, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, predict_classes),
        #     )

        self.time_series = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return {"predictions": self.clf(x), "time_series": self.time_series(x)}


class RegressorsMCDropout(nn.Module):
    def __init__(self, out_channels, hidden_dim, predict_classes, dropout_rate=0.2):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(p=dropout_rate),
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
            "training stage 1 finetune",
            "training stage 2 from embedding",
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

        if config.regressor_version == "mc_dropout":
            print("INFO: Using Regressors with MC Dropout for new model.")
            self.stage_1_regressors = RegressorsMCDropout(
                config.out_channels,
                config.stage_1_clf_hidden_dim,
                config.stage1_predict_classes,
                dropout_rate=config.dropout_rate,
            )
            # 同样为stage 2也进行替换
            self.stage_2_regressors = RegressorsMCDropout(
                config.out_channels,
                config.stage_2_clf_hidden_dim,
                config.stage2_predict_classes,
                dropout_rate=config.dropout_rate,
            )
        else:
            # 如果config中没有指定版本，或指定为legacy，则使用原始类
            print("INFO: Using original Regressors for backward compatibility.")
            self.stage_1_regressors = Regressors(
                config.out_channels,
                config.stage_1_clf_hidden_dim,
                config.stage1_predict_classes,
            )
            self.stage_2_regressors = Regressors(
                config.out_channels,
                config.stage_2_clf_hidden_dim,
                config.stage2_predict_classes,
            )

        # self.stage_1_regressors = Regressors(
        #     config.out_channels,
        #     config.stage_1_clf_hidden_dim,
        #     config.stage1_predict_classes,
        # )

        # stage 2 modules
        self.stage_2_encoder_blocks = nn.ModuleList(
            [
                SelfAttention(config.out_channels, config.n_head)
                for i in range(config.stage_2_transformer_layers)
            ]
        )

        # self.stage_2_encoder_blocks = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=config.out_channels,
        #         nhead=config.n_head,
        #         dim_feedforward=config.out_channels*2,
        #         dropout=0.1,
        #         activation="gelu",
        #     ),
        #     num_layers=config.stage_2_transformer_layers,
        # )

        self.stage_2_decoder_blocks = nn.ModuleList(
            [
                SelfAttention(config.out_channels, config.n_head)
                for i in range(config.stage_2_transformer_layers)
            ]
        )

        # self.stage_2_decoder_blocks = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=config.out_channels,
        #         nhead=config.n_head,
        #         dim_feedforward=config.out_channels*2,
        #         dropout=0.1,
        #         activation="gelu",
        #     ),
        #     num_layers=config.stage_2_transformer_layers,
        # )

        # self.stage_2_regressors = Regressors(
        #     config.out_channels,
        #     config.stage_2_clf_hidden_dim,
        #     config.stage2_predict_classes,
        # )

        # training utils
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.kl = nn.KLDivLoss(reduction="none")
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")
        self.cross_entropy_mutation = nn.CrossEntropyLoss(
            weight=torch.tensor([0.1, 1.0, 1.0])
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.last_train_step = 0

        self.pearson = torchmetrics.PearsonCorrCoef()

    # deprecated
    def stage1_forward_old(self, input_dict, masks=None):
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
                    q = input_dict["aligned_" + i]["seq_t"]
                    if q.dim() == 1:
                        q = q[None, :].repeat(batchsize, 1)

                    q = self.stage_1_embed(q)
                    x += q
            # print(x.shape, stage_1_ori_embeds[i].shape)
            x = self.stage_1_reconstructor(x)
            stage_1_logits[i] = x["aa_logits"]
            stage_1_aa_clf[i] = x["predict_logits"]

        return stage_1_embeds, stage_1_logits, stage_1_predicts, stage_1_aa_clf

    def stage1_forward(self, input_dict, masks=None):
        stage_1_embeds = {}
        stage_1_logits = {}
        stage_1_predicts = {}
        stage_1_aa_clf = {}
        for i in input_dict:
            if i not in self.prots:
                continue

            x = self.esm_model(input_dict[i])
            embed = self.stage_1_bottleneck(x)
            batchsize, length, channels = embed.shape

            # [CLS] token embedding for protein-wise prediction
            protein_embed = embed[:, 0]
            stage_1_embeds[i] = protein_embed

            predicts = self.stage_1_regressors(protein_embed)
            if self.stage == "inference":
                # predicts.pop("time_series")
                predicts.pop("fabricated")
            stage_1_predicts[i] = predicts

            x_recon = protein_embed[:, None].repeat(1, length, 1)

            if (
                self.config.teaching_force > 0.0
                and random.random() < self.config.teaching_force
            ):
                if "aligned_" + i in input_dict:
                    aligned_data = input_dict["aligned_" + i]

                    # 判断其是字典还是张量
                    if isinstance(aligned_data, dict):
                        # 如果是esm3的多模态字典, 则从字典中提取 'seq_t' 张量
                        seq_tensor = aligned_data["seq_t"]
                    else:
                        seq_tensor = aligned_data

                    if seq_tensor.dim() == 1:
                        # unsqueeze and repeat to match batch size
                        seq_tensor = seq_tensor.unsqueeze(0).repeat(batchsize, 1)

                    seq_embedded = self.stage_1_embed(seq_tensor)

                    # 确保维度匹配后相加
                    if seq_embedded.shape == x_recon.shape:
                        x_recon += seq_embedded

            x_output = self.stage_1_reconstructor(x_recon)
            stage_1_logits[i] = x_output["aa_logits"]
            stage_1_aa_clf[i] = x_output["predict_logits"]

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
        self,
        input_dict=None,
        stage_1_masks=None,
        stage_2_masks=None,
        only_stage_1=False,
        stage_1_embeds=None,
    ):
        # if stage_1_embeds is None:
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
            if "from embedding" in self.stage:
                stage_1_logits, stage_1_predicts, stage_1_aa_clf = None, None, None
            else:
                with torch.no_grad():
                    stage_1_embeds, stage_1_logits, stage_1_predicts, stage_1_aa_clf = (
                        self.stage1_forward(input_dict, stage_1_masks)
                    )
        # else: # 如果 stage_1_embeds 不为 None, 则 stage_1_logits 等为 None
        #     stage_1_logits, stage_1_predicts, stage_1_aa_clf = None, None, None

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
        prediction_losses = {}

        if "label" not in input_dict or self.config.stage1_predict_classes == 0:
            return prediction_losses

        for protein_name in output.S1Predicts:
            if "predictions" not in output.S1Predicts[protein_name]:
                # raise RuntimeError(
                #     f"output.S1Predicts中无predictions, 只有：'{output.S1Predicts}'"
                # )
                continue

            predicted_vector = output.S1Predicts[protein_name]["predictions"][0]
            true_vector = torch.cat(list(input_dict["label"].values()))
            true_vector = true_vector.to(dtype=predicted_vector.dtype)

            if true_vector.shape != predicted_vector.shape:
                raise RuntimeError(
                    f"无法匹配蛋白质'{protein_name}'的标签和预测形状。"
                    f"标签形状: {true_vector.shape}, 预测形状: {predicted_vector.shape}"
                )

            loss = self.mse(predicted_vector, true_vector)
            # print(true_vector)
            # loss = self.mse(predicted_vector, torch.zeros_like(predicted_vector))
            prediction_losses[protein_name] = loss

        # print(prediction_losses)

        return prediction_losses

    def fitness_losses(self, output: VESMOutputs, input_dict):
        fitness_losses = {}
        # print("\ninput dict label:",input_dict["label"])
        # print("\noutput:",output.S1Predicts)

        # 检查 'label' 是否存在于输入数据中
        if "label" not in input_dict:
            return fitness_losses

        # 遍历每个蛋白质的预测结果
        for i in output.S1Predicts:
            if (
                isinstance(input_dict["label"], dict)
                and "fitness_score" in input_dict["label"]
                and "predictions" in output.S1Predicts[i]
            ):
                # 提取真实值和预测值
                true_score = input_dict["label"]["fitness_score"].float()
                predicted_score = output.S1Predicts[i]["predictions"]

                # 假设 true_score 是一个标量或只有一个元素的张量
                if not isinstance(true_score, torch.Tensor):
                    true_score = torch.tensor(
                        true_score,
                        device=predicted_score.device,
                        dtype=predicted_score.dtype,
                    )

                # 确保 true_score 和 predicted_score 形状可以计算MSE
                if true_score.dim() == 0:
                    true_score = true_score.unsqueeze(0)
                if true_score.shape != predicted_score.shape:
                    true_score = true_score.view_as(predicted_score)

                # 计算MSE损失
                loss = self.mse(predicted_score, true_score)
                fitness_losses[i] = loss
                # print("fitness_losses:",fitness_losses)

        return fitness_losses

    def stage1_prediction_loss_old(self, output: VESMOutputs, input_dict):
        stage_1_prediction_loss = {}
        if "label" not in input_dict or self.config.stage1_predict_classes == 0:
            return stage_1_prediction_loss

        # xiugai
        # input_dict you labels, cong zhong du qu gai zhi,
        # output zhong you S1Predicts 中有predictions中有预测的值， 两者算一个mse返回
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

    def stage1_logit_loss_old(self, output: VESMOutputs, input_dict):
        stage_1_logitLosses = {}
        if "ori_seq" not in input_dict:
            return stage_1_logitLosses
        for i in output.S1Logits:
            s1 = output.S1Logits[i]
            if "ori_seq_kl" in input_dict:
                # use soft labels
                s1 = s1.log_softmax(dim=-1)
                l1 = input_dict["ori_seq_kl"][i]
                loss = self.kl(s1, l1).sum(-1)
            else:
                s1 = s1.view(-1, s1.shape[-1])
                l1 = input_dict["ori_seq"][i]["seq_t"].flatten()
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

    def stage1_logit_loss(self, output: VESMOutputs, input_dict):
        stage_1_logitLosses = {}
        if "ori_seq" not in input_dict:
            return stage_1_logitLosses

        for i in output.S1Logits:  # i 是蛋白质名称，如 'S'
            s1 = output.S1Logits[i]
            s1 = s1.view(-1, s1.shape[-1])

            ori_seq_data = input_dict["ori_seq"][i]
            # 判断其是字典还是张量
            if isinstance(ori_seq_data, dict):
                # 如果是esm3的多模态字典, 则从字典中提取 'seq_t' 张量
                l1_tensor = ori_seq_data["seq_t"]
            else:
                l1_tensor = ori_seq_data
            l1 = l1_tensor.flatten()
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
            # 避免除以零的错误
            if mask.sum() > 0:
                loss = loss.sum() / mask.sum()
            else:
                loss = loss.sum()  # 如果没有可计算损失的token，则不进行平均
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
        prediction_losses = None

        if "label" not in input_dict or "predictions" not in output.S2Predicts:
            return prediction_losses

        predicted_vector = output.S2Predicts["predictions"][0]
        true_vector = torch.cat(list(input_dict["label"].values()))
        true_vector = true_vector.to(dtype=predicted_vector.dtype)

        if true_vector.shape != predicted_vector.shape:
            raise RuntimeError(
                f"无法匹配stage2嵌入的标签和预测形状。"
                f"标签形状: {true_vector.shape}, 预测形状: {predicted_vector.shape}"
            )

        loss = self.mse(predicted_vector, true_vector)

        return loss

    def stage2_prediction_loss_old(self, output: VESMOutputs, input_dict):
        stage_2_prediction_loss = None
        if "label" not in input_dict:
            return stage_2_prediction_loss
        t = input_dict["label"]["stage2"].float()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if self.config.stage_2_predictLosses is not None:
            loss = self.config.stage_2_predictLosses(
                output.S2Predicts["predictions"], t
            )
        else:
            loss = self.bce(
                output.S2Predicts["predictions"],
                t,
            )
        return loss

    def getLoss(self, input_dict1, input_dict2=None):

        if "from embedding" in self.stage:
            output1 = self.forward(
                stage_1_embeds=input_dict1["input"],
                stage_2_masks=input_dict1.get("stage_2_masks", None),
            )
        else:
            output1 = self.forward(
                input_dict1["input"],
                input_dict1.get("stage_1_masks", None),
                input_dict1.get("stage_2_masks", None),
            )

        if input_dict2 is not None:
            if "from embedding" in self.stage:
                output2 = self.forward(
                    stage_1_embeds=input_dict2["input"],
                    stage_2_masks=input_dict2.get("stage_2_masks", None),
                )
            else:
                output2 = self.forward(
                    input_dict2["input"],
                    input_dict2.get("stage_1_masks", None),
                    input_dict2.get("stage_2_masks", None),
                )
        else:
            output2 = None

        # print(input_dict1)
        # output1 = self.forward(
        #     input_dict1["input"],
        #     input_dict1.get("stage_1_masks", None),
        #     input_dict1.get("stage_2_masks", None),
        # )
        # if input_dict2 is not None:
        #     output2 = self.forward(
        #         input_dict2["input"],
        #         input_dict2.get("stage_1_masks", None),
        #         input_dict2.get("stage_2_masks", None),
        #     )
        # else:
        #     output2 = None

        # stage 1 losses
        stage_1_logitLosses = {}
        stage_1_predictLosses = {}
        stage_1_predictAALosses = {}

        # 不从emb训练才执行stage1 loss
        # if "stage_1_embeds" not in input_dict1:
        if "from embedding" not in self.stage:
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

            # xiugai
            # zeng jia liang ge mse loss shang qu
            if "label" in input_dict1:
                stage_1_predict_loss = self.stage1_prediction_loss(output1, input_dict1)
            else:
                stage_1_predict_loss = {}

            for i in stage_1_predict_loss:
                stage_1_predictLosses[i + "_predicted_1"] = stage_1_predict_loss[i]

            if output2 is not None:
                if "label" in input_dict2:
                    stage_1_predict_loss = self.stage1_prediction_loss(
                        output2, input_dict2
                    )
                else:
                    stage_1_predict_loss = {}
                for i in stage_1_predict_loss:
                    stage_1_predictLosses[i + "_predicted_2"] = stage_1_predict_loss[i]

            if output2 is not None:
                s = self.stage1_time_series_loss(output1, output2)
                for i in s:
                    stage_1_predictLosses[i + "_time_series"] = s[i]

        # print("stage_1_predictLosses:",stage_1_predictLosses)

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
        # else:
        #     stage_2_predictLosses = None

        # 新增stage2 predictions
        if "label" in input_dict1:
            stage_2_predictLosses["predicted_1"] = self.stage2_prediction_loss(
                output1, input_dict1
            )
        else:
            stage_2_predictLosses["predicted_1"] = {}

        if output2 is not None:
            if "label" in input_dict2:
                stage_2_predictLosses["predicted_2"] = self.stage2_prediction_loss(
                    output2, input_dict2
                )
            else:
                stage_2_predictLosses["predicted_2"] = {}

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
            # 如果是从 embedding 开始训练， S1 部分的 loss 就是0
            # # if "stage_1_embeds" in input_dict1:

            #     loss1 = sum([i for i in loss.S2ReconstructLosses.values()])
            #     predict_loss_val = loss.S2PredictsLoss
            #     if isinstance(predict_loss_val, dict):
            #         loss3 = sum(predict_loss_val.values())
            #     else:
            #         loss3 = predict_loss_val if predict_loss_val is not None else 0.0

            #     loss_val = (
            #         loss1 * self.config.stage_2_recosntruct_weight
            #         + loss3 * self.config.stage_2_regressor_weight
            #     )
            #     d = {
            #         "S2ReconstructLosses": loss1.detach().cpu(),
            #         "S2PredictsLoss": loss3.detach().cpu() if torch.is_tensor(loss3) else torch.tensor(loss3),
            #         "loss": loss_val.detach().cpu(),
            #     }
            #     return loss_val, d
            # else:
            loss1 = sum([i for i in loss.S2ReconstructLosses.values()])

            # loss3 = loss.S2PredictsLoss
            predict_loss_val = loss.S2PredictsLoss

            # 检查 S2PredictsLoss 是否是字典
            if isinstance(predict_loss_val, dict):
                loss3 = sum(predict_loss_val.values())
            else:
                loss3 = predict_loss_val if predict_loss_val is not None else 0.0

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

        if self.stage == "training stage 2 from embedding":
            # 如果是从 embedding 开始训练， S1 部分的 loss 就是0

            loss1 = sum([i for i in loss.S2ReconstructLosses.values()])
            predict_loss_val = loss.S2PredictsLoss
            if isinstance(predict_loss_val, dict):
                loss3 = sum(predict_loss_val.values())
            else:
                loss3 = predict_loss_val if predict_loss_val is not None else 0.0

            loss_val = (
                loss1 * self.config.stage_2_recosntruct_weight
                + loss3 * self.config.stage_2_regressor_weight
            )
            d = {
                "S2ReconstructLosses": loss1.detach().cpu(),
                "S2PredictsLoss": (
                    loss3.detach().cpu()
                    if torch.is_tensor(loss3)
                    else torch.tensor(loss3)
                ),
                "loss": loss_val.detach().cpu(),
            }
            return loss_val, d

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
        with torch.no_grad():
            if "from embedding" in self.stage:
                output1 = self.forward(
                    stage_1_embeds=input_dict1["input"],
                    stage_2_masks=input_dict1.get("stage_2_masks", None),
                )
            else:
                output1 = self.forward(
                    input_dict1["input"],
                    input_dict1.get("stage_1_masks", None),
                    input_dict1.get("stage_2_masks", None),
                )

            if input_dict2 is not None:
                if "from embedding" in self.stage:
                    output2 = self.forward(
                        stage_1_embeds=input_dict2["input"],
                        stage_2_masks=input_dict2.get("stage_2_masks", None),
                    )
                else:
                    output2 = self.forward(
                        input_dict2["input"],
                        input_dict2.get("stage_1_masks", None),
                        input_dict2.get("stage_2_masks", None),
                    )
            else:
                output2 = None

        dp = {}

        if input_dict1["label"]:

            dp["label_key"] = list(input_dict1["label"].keys())
            dp["output1_x"] = {}
            true_vector = torch.cat(list(input_dict1["label"].values()))
            for protein_name in output1.S1Predicts:
                predicted_vector = output1.S1Predicts[protein_name]["predictions"][
                    0
                ].detach()
                dp["output1_x"][protein_name] = predicted_vector
            dp["output1_y"] = true_vector
            if output2 is not None:
                dp["output2_x"] = {}
                true_vector = torch.cat(list(input_dict2["label"].values()))
                for protein_name in output2.S1Predicts:
                    predicted_vector = output2.S1Predicts[protein_name]["predictions"][
                        0
                    ].detach()
                    dp["output2_x"][protein_name] = predicted_vector
                dp["output2_y"] = true_vector
            else:
                dp["output2_x"] = None
                dp["output2_y"] = None

        loss, d = self._common_training_step(input_dict1, input_dict2)

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
        x = []
        y = []
        # print(self.validation_step_outputs)
        if "output1_x" in self.validation_step_outputs[0]:
            for i in self.validation_step_outputs:
                t = i.pop("output2_x")
                t = i.pop("output1_x")
                x.append(t["S"])
                t = i.pop("output2_y")
                t = i.pop("output1_y")
                y.append(t)
                key = i.pop("label_key")

            x = torch.stack(x)
            y = torch.stack(y)

            # print("key:",key)
            print("x:", x)
            # print("y:",y)
            p_sum = 0
            for i in range(len(key)):
                p = self.pearson(x[:, i], y[:, i])
                self.pearson.reset()
                p_sum += p
                self.log(f"validation_pearson_{key[i]}", p)

            # self.pearson.reset()
            self.log(f"validation_pearson", p_sum)

            # p = self.pearson(x[:, 1], y[:, 1])
            # self.pearson.reset()
            # self.log("validation_pearson_1", p)

            # p = self.pearson(x[:, 2], y[:, 0])
            # self.pearson.reset()
            # self.log("validation_pearson_0", p)

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

        if "finetune" in self.stage:
            for i, j in self.named_parameters():
                freeze = True
                if "1" in self.stage:
                    if "1_regressors" in i:
                        freeze = False
                if "2" in self.stage:
                    if "2_regressors" in i:
                        freeze = False
                if freeze:
                    j.requires_grad_(False)
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

    # def on_save_checkpoint(self, checkpoint):
    #     backbones = []
    #     for i in checkpoint["state_dict"]:
    #         if "esm" in i and "lora" not in i:
    #             backbones.append(i)
    #     for i in backbones:
    #         del checkpoint["state_dict"][i]

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i, j in self.named_parameters():
            if ("esm" in i and not j.requires_grad) and ("lora" not in i):
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]

    # 添加一个方法来激活 MC Dropout
    def enable_mc_dropout(self):
        """在所有模块中激活 Dropout 层"""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # 将 Dropout 层设置为训练模式

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


class MyESM(nn.Module):
    """A lightweight Transformer encoder with rotary positional embeddings.

    Specs:
      - vocab_size: 33
      - d_model: 512
      - nhead: 16 (head_dim=32)
      - num_layers: 5

    Output contract:
      forward(sequence_tokens: LongTensor[B, L]) -> object with attribute `.embeddings`
      where `.embeddings` has shape (B, L, 512).
    """

    def __init__(
        self,
        vocab_size: int = 33,
        d_model: int = 512,
        nhead: int = 16,
        num_layers: int = 5,
        pad_idx: int = 0,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.pad_idx = pad_idx

        # Token embedding only; rotary embeddings are applied inside attention blocks
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout_p)

        # Stack of SelfAttention blocks which already includes RoPE and FFN
        self.blocks = nn.ModuleList(
            [SelfAttention(d_model, nhead) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

        # class _Output:
        #     def __init__(self, embeddings: torch.Tensor):
        #         self.embeddings = embeddings

        # self._Output = _Output

    def forward(self, sequence_tokens: torch.Tensor):
        # Accept (L,) or (B, L) tokens in [0, vocab_size)
        if sequence_tokens.dim() == 1:
            sequence_tokens = sequence_tokens.unsqueeze(0)

        x = self.tok_embed(sequence_tokens)
        x = self.dropout(x)

        # Pass through rotary attention blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.final_ln(x)
        return x


class MyESM_shabi(nn.Module):
    def __init__(self):
        super().__init__()
        # Provide a simple Transformer encoder-based ESM-like backbone.
        # Defaults: 33 vocab size, 6 layers, d_model=512, nhead=16.
        # Keep signature compatible; allow overriding via attributes if needed.
        self.vocab_size = 33
        self.d_model = 512
        self.nhead = 16
        self.num_layers = 6
        self.max_len = 2048
        self.pad_idx = 0
        self.dropout_p = 0.1

        self.tok_embed = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=self.pad_idx
        )
        self.pos_embed = nn.Embedding(self.max_len, self.d_model)
        self.dropout = nn.Dropout(self.dropout_p)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 2,
            dropout=self.dropout_p,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.final_ln = nn.LayerNorm(self.d_model)

        # Small holder to mimic ESM-style outputs
        # class _Output:
        #     def __init__(self, embeddings: torch.Tensor):
        #         self.embeddings = embeddings

        # self._Output = _Output

    def forward(self, sequence_tokens: torch.Tensor):
        # Accept shape (L,) or (B, L) of token ids in [0, vocab_size)
        if sequence_tokens.dim() == 1:
            sequence_tokens = sequence_tokens.unsqueeze(0)

        bsz, seqlen = sequence_tokens.size()
        device = sequence_tokens.device

        # Build embeddings
        pos_ids = torch.arange(seqlen, device=device).unsqueeze(0).expand(bsz, seqlen)
        x = self.tok_embed(sequence_tokens) + self.pos_embed(pos_ids)
        x = self.dropout(x)

        # Padding mask: True for pads to be ignored by attention
        key_padding_mask = sequence_tokens.eq(self.pad_idx)

        # Encode
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.final_ln(x)

        # Return an object with `.embeddings` of shape (B, L, C)
        return x


# 无bottleneck
class SimpleESM(L.LightningModule):
    def __init__(self, esm_model, config: VESMConfig):
        super().__init__()

        # print("model at stage:", stage)

        self.config = config
        self.prots = config.prots

        # print("======",config.track)
        self.esm_model = ESMModule(esm_model, config.esm_model_type, config.track)

        self.aa_clf = nn.Sequential(
            nn.Linear(config.esm_model_channels, config.esm_model_channels // 2),
            nn.GELU(),
            nn.Linear(config.esm_model_channels // 2, config.aa_counts),
        )

        # self.stage_1_regressors = Regressors(
        #        config.esm_model_channels,
        #        # 1275,
        #        config.stage_1_clf_hidden_dim,
        #        config.stage1_predict_classes,
        #    )
        self.regressor = nn.Sequential(
            nn.Linear(config.esm_model_channels, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.aa_predict = nn.Sequential(
            nn.Linear(config.esm_model_channels, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.return_classifier_results = config.stage1_predict_classes > 0

        self.cross_entropy = nn.CrossEntropyLoss(reduce="None")
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1000.0)))
        self.config = config
        self.prots = config.prots
        self.mse = nn.MSELoss()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.pearson = torchmetrics.PearsonCorrCoef()
        self.cnts = 0

    def forward(self, input_dict, position=None, thres=1.0):
        embeds = {}
        logits = {}
        stage_1_predicts = {}
        for i in input_dict:
            if i not in self.prots:
                continue
            # print(input_dict[i])
            x = self.esm_model(input_dict[i])
            embeds[i] = x
            logits[i] = self.aa_clf(x)
            if self.return_classifier_results:
                # print(input_dict)
                aa_logit = self.aa_predict(x)
                aa_logit = aa_logit.squeeze(-1)
                weight = torch.softmax(aa_logit, 1)
                if position is None:
                    predicts = self.regressor(x[:, 0, :])
                else:
                    pooled = torch.einsum("blc,bl->bc", x, weight.detach())
                    predicts = self.regressor(pooled)
                    # if torch.rand(1) > thres:
                    #     predicts = self.regressor(x[:, position, :])
                    # else:
                    #     predicts = self.regressor(x[:, 0, :])
                stage_1_predicts[i] = predicts

        if self.return_classifier_results:
            #    stage_1_predicts = {}
            #    for i in input_dict:
            #        if i not in self.prots:
            #            continue
            #        embed = embeds[i]
            #       embed = torch.mean(embed, axis=1)
            #       predicts = self.stage_1_regressors(embed)
            #        stage_1_predicts[i] = predicts
            return embeds, logits, stage_1_predicts, aa_logit

        return embeds, logits

    def _common_training_step(self, batch, outputs=None, thres=1.0):
        # print(batch)

        if self.return_classifier_results:
            position = [int(i[1:-1]) for i in batch["meta"]["id"]]
            position = torch.tensor(position).to(self.device) - 1
            position.requires_grad_(False)
            # print(batch["meta"], position)
            # position = None
            output = self.forward(batch["input"], position=position, thres=thres)
            embeds, logits, stage_1_predicts, aa_logit = output
        else:
            output = self.forward(batch["input"])
            embeds, logits = output
        total_loss = 0.0
        for i in logits:
            s1 = logits[i]
            s1 = s1.view(-1, s1.shape[-1])

            if isinstance(batch["ori_seq"][i], dict):
                # 如果是字典，则提取 'seq_t'
                ori_seq = batch["ori_seq"][i]["seq_t"]
            else:
                ori_seq = batch["ori_seq"][i]

            l1 = ori_seq.flatten()
            loss = self.cross_entropy(s1, l1)
            if "stage_1_masks" in batch and batch["stage_1_masks"] is not None:
                mask = batch["stage_1_masks"][i].flatten().float()
                mask = 1 - mask
                # print("mask", mask)
                mask[mask < 0] = -self.config.stage_1_masked_weight
                mask += self.config.stage_1_masked_weight
            else:
                mask = torch.ones_like(loss)
            mask[0] = 0.0
            loss = loss * mask
            loss = loss.sum() / mask.sum()
            total_loss += loss
        if self.return_classifier_results:
            mse_loss = 0
            bce_loss = 0
            for i in logits:
                predicted_vector = stage_1_predicts[i][0]
                gth = torch.zeros_like(aa_logit)
                gth[:, position] = 1.0
                true_vector = torch.cat(list(batch["label"].values()))
                true_vector = true_vector.to(dtype=predicted_vector.dtype)
                if outputs is not None:
                    outputs.append(
                        {
                            "true_vector": true_vector.detach(),
                            "predicted_vector": predicted_vector.detach(),
                        }
                    )
                loss = self.mse(predicted_vector, true_vector)
                mse_loss += loss
                bce_loss += self.bce(aa_logit, gth)
            # print("mse_loss", mse_loss)
            self.log("MSE_loss", mse_loss)
            self.log("BCE_loss", bce_loss)
            return mse_loss * 0.5 + bce_loss, embeds
            total_loss += mse_loss
        return total_loss, embeds

    def training_step(self, batch, batch_idx):
        loss, d = self._common_training_step(batch, None, thres=-0.5)
        self.log("training_loss", loss.detach().cpu(), prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) > 0:
            x = [i["predicted_vector"] for i in self.validation_step_outputs]
            y = [i["true_vector"] for i in self.validation_step_outputs]
            x = torch.stack(x).squeeze()
            y = torch.stack(y).squeeze()
            # self.cnts += 1
            # if self.cnts % 200 == 0:
            print(x)
            p = self.pearson(x, y)
            self.pearson.reset()
            self.log("validation_pearson", p)
            self.validation_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        loss, d = self._common_training_step(
            batch, self.validation_step_outputs, thres=-1.0
        )
        self.log("validation_loss", loss.detach().cpu(), prog_bar=True)
        return loss

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
        else:
            return torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.parameters()),
                self.config.lr,
            )

    # def on_save_checkpoint(self, checkpoint):
    #     backbones = []
    #     for i in checkpoint["state_dict"]:
    #         if "esm" in i and "lora" not in i:
    #             backbones.append(i)
    #     for i in backbones:
    #         del checkpoint["state_dict"][i]

    def on_save_checkpoint(self, checkpoint):
        backbones = []
        for i, j in self.named_parameters():
            if ("esm" in i and not j.requires_grad) and ("lora" not in i):
                backbones.append(i)
        for i in backbones:
            del checkpoint["state_dict"][i]


class ESMModule(nn.Module):
    def __init__(self, esm_model, esm_model_type, track=None):
        super().__init__()
        self.esm_model = esm_model
        self.esm_model_type = esm_model_type
        self.num_layers = 33
        if track is not None:
            self.track = track
            print(f"only use {self.track} for training")
        else:
            self.track = ["seq_t", "structure_t", "ss8_t", "sasa_t"]

    def forward(self, input_dict):
        if self.esm_model_type == "esm3":

            for i in ["seq_t", "structure_t", "ss8_t", "sasa_t"]:
                if i not in input_dict:
                    input_dict[i] = None
                elif i not in self.track:
                    input_dict[i] = None
                else:
                    if input_dict[i] is None:
                        continue
                    if len(input_dict[i].size()) == 1:
                        input_dict[i] = input_dict[i].unsqueeze(0)

            # for i in self.track:
            #     if i not in input_dict:
            #         input_dict[i] = None
            #     else:
            #         if input_dict[i] is None:
            #             continue
            #         if len(input_dict[i].size()) == 1:
            #             input_dict[i] = input_dict[i].unsqueeze(0)

            # print("seq_t: ", input_dict["seq_t"])
            # print("\nseqstructure_t: ", input_dict["structure_t"])

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

            t = input_dict
            # print(input_dict)
            if len(t.size()) == 1:
                t = t.unsqueeze(0)
            representations = self.esm_model(t, repr_layers=[self.num_layers])

            x = representations["representations"][self.num_layers]
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

        if self.esm_model_type == "dummy":
            t = input_dict

            if len(t.size()) == 1:
                t = t.unsqueeze(0)

            representations = self.esm_model(
                sequence_tokens=t,
            )

            x = representations
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


# deprecated
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
