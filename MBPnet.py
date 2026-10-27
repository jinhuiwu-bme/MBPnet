"""MBPnet: Multi-scale Bottleneck Prototype Network for cognitive load assessment
with ECG and EEG signals.

Components:
    MTE: Multi-scale Temporal Encoding
    ChannelAggregation: EEG channel aggregation
    BRF: Bottleneck-based Reliability-aware Fusion
    PAR: Prototype Alignment Radius strategy
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttentionPooling(nn.Module):
    """Attention pooling: [B, N, D] -> [B, D]."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        weights = torch.softmax(self.attn(x), dim=1)
        return (x * weights).sum(dim=1)


# =============================================================================
# MTE: Multi-scale Temporal Encoding
# =============================================================================


class ConvNormAct1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 1,
        padding: int | None = None,
        groups: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class MultiScaleTemporalMixing(nn.Module):
    """Multi-scale aggregation: average-pooling decomposition and coarse-to-fine residual mixing."""

    def __init__(self, d_model: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.projection_3_to_2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )
        self.projection_2_to_1 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        tau_1 = x                                                   # [B, D, T]
        tau_2 = F.avg_pool1d(tau_1, kernel_size=2, stride=2)        # [B, D, T/2]
        tau_3 = F.avg_pool1d(tau_2, kernel_size=2, stride=2)        # [B, D, T/4]

        xi_3 = tau_3
        xi_2 = tau_2 + self.projection_3_to_2(
            F.interpolate(xi_3, size=tau_2.shape[-1], mode="linear", align_corners=False)
        )
        xi_1 = tau_1 + self.projection_2_to_1(
            F.interpolate(xi_2, size=tau_1.shape[-1], mode="linear", align_corners=False)
        )
        return xi_1


class PatchTemporalTransformer(nn.Module):
    """Temporal dependency modeling: patch embedding and a pre-norm Transformer block."""

    def __init__(
        self,
        d_model: int = 64,
        patch_size: int = 32,
        stride: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_projection = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
            bias=False,
        )
        self.norm_1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_projection(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]

        normalized = self.norm_1(x)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
        )
        x = x + attention_output
        x = x + self.ffn(self.norm_2(x))
        return x


class MTE(nn.Module):
    """Multi-scale Temporal Encoding for one signal channel: [B, 1, T] -> [B, N, D]."""

    def __init__(
        self,
        d_model: int = 64,
        patch_size: int = 32,
        stride: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvNormAct1D(1, d_model // 2, kernel_size=15, dropout=dropout),
            ConvNormAct1D(d_model // 2, d_model, kernel_size=7, dropout=dropout),
        )
        self.multi_scale_mixing = MultiScaleTemporalMixing(
            d_model=d_model,
            dropout=dropout,
        )
        self.patch_temporal_transformer = PatchTemporalTransformer(
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            num_heads=4,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.multi_scale_mixing(x)
        return self.patch_temporal_transformer(x)


# =============================================================================
# Channel Aggregation
# =============================================================================


class ChannelAggregation(nn.Module):
    """Aggregate the Fp1, Fp2, and Fp1-Fp2 token sequences: 3 x [B, N, D] -> [B, N, D]."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.pool = AttentionPooling(d_model)

        self.channel_gate_1 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.channel_gate_2 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )
        self.channel_gate_3 = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, 1),
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(3 * d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm1d(d_model),
            nn.SiLU(),
        )

    def forward(self, channel_tokens: List[Tensor]) -> Tensor:
        if len(channel_tokens) != 3:
            raise ValueError(
                f"ChannelAggregation expects exactly three EEG channels, got {len(channel_tokens)}."
            )

        channel_1, channel_2, channel_3 = channel_tokens

        summary_1 = self.pool(channel_1)
        summary_2 = self.pool(channel_2)
        summary_3 = self.pool(channel_3)

        gate_1 = torch.sigmoid(self.channel_gate_1(summary_1)).unsqueeze(1)
        gate_2 = torch.sigmoid(self.channel_gate_2(summary_2)).unsqueeze(1)
        gate_3 = torch.sigmoid(self.channel_gate_3(summary_3)).unsqueeze(1)

        gated_1 = gate_1 * channel_1
        gated_2 = gate_2 * channel_2
        gated_3 = gate_3 * channel_3

        fused = torch.cat([gated_1, gated_2, gated_3], dim=-1)  # [B, N, 3D]
        fused = fused.transpose(1, 2)  # [B, 3D, N]
        fused = self.fusion(fused)  # [B, D, N]
        return fused.transpose(1, 2)  # [B, N, D]


# =============================================================================
# BRF: Bottleneck-based Reliability-aware Fusion
# =============================================================================


class BottleneckCrossModalFusion(nn.Module):
    """Bottleneck-based cross-modal fusion with learnable bottleneck tokens."""

    def __init__(
        self,
        d_model: int = 64,
        num_bottlenecks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_bottlenecks = num_bottlenecks
        self.bottleneck_tokens = nn.Parameter(
            torch.randn(1, num_bottlenecks, d_model) * 0.02
        )

        self.eeg_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ecg_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.pool = AttentionPooling(d_model)

    def forward(self, eeg_tokens: Tensor, ecg_tokens: Tensor) -> Tensor:
        batch_size = eeg_tokens.size(0)
        bottlenecks = self.bottleneck_tokens.expand(batch_size, -1, -1)

        eeg_bottlenecks, _ = self.eeg_attention(
            bottlenecks,
            eeg_tokens,
            eeg_tokens,
        )
        ecg_bottlenecks, _ = self.ecg_attention(
            bottlenecks,
            ecg_tokens,
            ecg_tokens,
        )

        fused_bottlenecks = self.norm(eeg_bottlenecks + ecg_bottlenecks)
        fused_bottlenecks = fused_bottlenecks + self.ffn(fused_bottlenecks)
        return self.pool(fused_bottlenecks)


class ReliabilityAwareGate(nn.Module):
    """Reliability-aware gating of modality-specific representations."""

    def __init__(self, d_model: int = 64) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model * 2),
        )

    def forward(
        self,
        eeg_representation: Tensor,
        ecg_representation: Tensor,
        cross_modal_representation: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        gate_logits = self.gate(
            torch.cat(
                [
                    eeg_representation,
                    ecg_representation,
                    cross_modal_representation,
                ],
                dim=1,
            )
        )
        eeg_gate, ecg_gate = torch.chunk(
            torch.sigmoid(gate_logits),
            chunks=2,
            dim=1,
        )

        fused_representation = (
            cross_modal_representation
            + eeg_gate * eeg_representation
            + ecg_gate * ecg_representation
        )
        return fused_representation, eeg_gate, ecg_gate


class BRF(nn.Module):
    """Bottleneck-based Reliability-aware Fusion.

    Inputs:
        eeg_tokens: [B, N_eeg, D]
        ecg_tokens: [B, N_ecg, D]
    Outputs:
        fused_representation, cross_modal_representation, eeg_representation,
        ecg_representation, eeg_gate, ecg_gate: each [B, D]
    """

    def __init__(
        self,
        d_model: int = 64,
        num_bottlenecks: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.unimodal_pool = AttentionPooling(d_model)
        self.cross_modal_fusion = BottleneckCrossModalFusion(
            d_model=d_model,
            num_bottlenecks=num_bottlenecks,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.reliability_gate = ReliabilityAwareGate(d_model=d_model)

    def forward(
        self,
        eeg_tokens: Tensor,
        ecg_tokens: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        eeg_representation = self.unimodal_pool(eeg_tokens)
        ecg_representation = self.unimodal_pool(ecg_tokens)
        cross_modal_representation = self.cross_modal_fusion(
            eeg_tokens,
            ecg_tokens,
        )
        fused_representation, eeg_gate, ecg_gate = self.reliability_gate(
            eeg_representation,
            ecg_representation,
            cross_modal_representation,
        )
        return (
            fused_representation,
            cross_modal_representation,
            eeg_representation,
            ecg_representation,
            eeg_gate,
            ecg_gate,
        )


# =============================================================================
# PAR: Prototype Alignment Radius strategy
# =============================================================================


PAR_LOSS_WEIGHT = 0.05
PAR_PULL_WEIGHT = 1.15
PAR_TASK_CENTER_WEIGHT = 0.10
PAR_PROTOTYPE_MOMENTUM = 0.993
PAR_MIN_COUNT = 5
PAR_RADIUS_INIT = 0.12
PAR_RADIUS_MIN = 0.03
PAR_RADIUS_MAX = 0.24
PAR_RADIUS_STD_ALPHA = 0.15
PAR_REFERENCE_ALPHA = 0.70


class GlobalPrototypeBank:
    """Global prototype and radius for each CL category, updated by EMA."""

    def __init__(
        self,
        num_classes: int = 3,
        feature_dim: int = 64,
        momentum: float = PAR_PROTOTYPE_MOMENTUM,
        min_count: int = PAR_MIN_COUNT,
        device: torch.device | str = "cuda",
    ) -> None:
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.min_count = min_count
        self.device = torch.device(device)

        self.prototypes = torch.zeros(
            num_classes,
            feature_dim,
            device=self.device,
        )
        self.radii = torch.full(
            (num_classes,),
            PAR_RADIUS_INIT,
            device=self.device,
        )
        self.initialized = torch.zeros(
            num_classes,
            dtype=torch.bool,
            device=self.device,
        )

    @torch.no_grad()
    def update(self, features: Tensor, labels: Tensor) -> None:
        if features.numel() == 0:
            return

        features = F.normalize(features.detach(), dim=1, eps=1e-6)

        for class_index in range(self.num_classes):
            class_mask = labels == class_index
            if class_mask.sum().item() < self.min_count:
                continue

            class_features = features[class_mask]
            batch_prototype = F.normalize(
                class_features.mean(dim=0),
                dim=0,
                eps=1e-6,
            )

            if not self.initialized[class_index]:
                self.prototypes[class_index] = batch_prototype
                self.initialized[class_index] = True
            else:
                updated_prototype = (
                    self.momentum * self.prototypes[class_index]
                    + (1.0 - self.momentum) * batch_prototype
                )
                updated_prototype = F.normalize(
                    updated_prototype,
                    dim=0,
                    eps=1e-6,
                )
                if torch.isnan(updated_prototype).any() or torch.isinf(
                    updated_prototype
                ).any():
                    continue
                self.prototypes[class_index] = updated_prototype

            distances = 1.0 - F.cosine_similarity(
                class_features,
                self.prototypes[class_index].unsqueeze(0).expand_as(class_features),
                dim=1,
            )
            batch_radius = distances.mean() + PAR_RADIUS_STD_ALPHA * distances.std(
                unbiased=False
            )
            batch_radius = torch.clamp(
                batch_radius,
                PAR_RADIUS_MIN,
                PAR_RADIUS_MAX,
            )
            if torch.isnan(batch_radius) or torch.isinf(batch_radius):
                continue

            self.radii[class_index] = (
                self.momentum * self.radii[class_index]
                + (1.0 - self.momentum) * batch_radius
            )
            self.radii[class_index] = torch.clamp(
                self.radii[class_index],
                PAR_RADIUS_MIN,
                PAR_RADIUS_MAX,
            )

    def get(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.prototypes, self.initialized, self.radii

    def state_dict(self) -> Dict[str, Tensor]:
        return {
            "prototypes": self.prototypes.detach().clone(),
            "radii": self.radii.detach().clone(),
            "initialized": self.initialized.detach().clone(),
        }

    def load_state_dict(self, state: Dict[str, Tensor]) -> None:
        self.prototypes.copy_(state["prototypes"].to(self.device))
        self.radii.copy_(state["radii"].to(self.device))
        self.initialized.copy_(state["initialized"].to(self.device))


@torch.no_grad()
def build_reference_prototypes(
    source_features: Tensor,
    source_labels: Tensor,
    prototype_bank: GlobalPrototypeBank,
    num_classes: int = 3,
    min_count: int = PAR_MIN_COUNT,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Reference prototypes and radii from global and mini-batch statistics."""

    global_prototypes, global_valid, global_radii = prototype_bank.get()

    reference_prototypes = global_prototypes.clone().detach()
    reference_valid = global_valid.clone().detach()
    reference_radii = global_radii.clone().detach()

    if source_features.numel() == 0:
        return reference_prototypes, reference_valid, reference_radii

    source_features = F.normalize(source_features.detach(), dim=1, eps=1e-6)

    for class_index in range(num_classes):
        class_mask = source_labels == class_index
        if class_mask.sum().item() < min_count:
            continue

        class_features = source_features[class_mask]
        local_prototype = F.normalize(
            class_features.mean(dim=0),
            dim=0,
            eps=1e-6,
        )

        local_distances = 1.0 - F.cosine_similarity(
            class_features,
            local_prototype.unsqueeze(0).expand_as(class_features),
            dim=1,
        )
        local_radius = (
            local_distances.mean()
            + PAR_RADIUS_STD_ALPHA * local_distances.std(unbiased=False)
        )
        local_radius = torch.clamp(
            local_radius,
            PAR_RADIUS_MIN,
            PAR_RADIUS_MAX,
        )

        if bool(global_valid[class_index].item()):
            fused_prototype = (
                PAR_REFERENCE_ALPHA * global_prototypes[class_index]
                + (1.0 - PAR_REFERENCE_ALPHA) * local_prototype
            )
            fused_prototype = F.normalize(
                fused_prototype,
                dim=0,
                eps=1e-6,
            )
            fused_radius = (
                PAR_REFERENCE_ALPHA * global_radii[class_index]
                + (1.0 - PAR_REFERENCE_ALPHA) * local_radius
            )
            fused_radius = torch.clamp(
                fused_radius,
                PAR_RADIUS_MIN,
                PAR_RADIUS_MAX,
            )

            reference_prototypes[class_index] = fused_prototype
            reference_radii[class_index] = fused_radius
        else:
            reference_prototypes[class_index] = local_prototype
            reference_radii[class_index] = local_radius

        reference_valid[class_index] = True

    return reference_prototypes, reference_valid, reference_radii


def sample_radius_alignment_loss(
    features: Tensor,
    labels: Tensor,
    prototypes: Tensor,
    valid_mask: Tensor,
    radii: Tensor,
) -> Tensor:
    """Sample-level alignment loss: penalizes only deviations beyond the class radius."""

    zero = features.sum() * 0.0
    if features.numel() == 0:
        return zero

    features = F.normalize(features, dim=1, eps=1e-6)
    loss = zero
    valid_class_count = 0

    for class_index in range(prototypes.size(0)):
        if not bool(valid_mask[class_index].item()):
            continue

        class_mask = labels == class_index
        if class_mask.sum().item() == 0:
            continue

        class_features = features[class_mask]
        class_prototype = prototypes[class_index].detach().unsqueeze(0).expand_as(
            class_features
        )
        class_radius = radii[class_index].detach().clamp(
            min=PAR_RADIUS_MIN,
            max=PAR_RADIUS_MAX,
        )

        distances = 1.0 - F.cosine_similarity(
            class_features,
            class_prototype,
            dim=1,
        )
        overflow = F.relu(distances - class_radius)
        loss = loss + overflow.pow(2).mean()
        valid_class_count += 1

    if valid_class_count > 0:
        loss = loss / valid_class_count
    return loss


def task_center_radius_alignment_loss(
    features: Tensor,
    labels: Tensor,
    tasks: Tensor,
    prototypes: Tensor,
    valid_mask: Tensor,
    radii: Tensor,
    min_count: int = PAR_MIN_COUNT,
) -> Tensor:
    """Task-level alignment loss: penalizes task-wise class centers beyond the class radius."""

    zero = features.sum() * 0.0
    if features.numel() == 0:
        return zero

    features = F.normalize(features, dim=1, eps=1e-6)
    loss = zero
    valid_center_count = 0
    unique_tasks = torch.unique(tasks)

    for class_index in range(prototypes.size(0)):
        if not bool(valid_mask[class_index].item()):
            continue

        class_prototype = prototypes[class_index].detach()
        class_radius = radii[class_index].detach().clamp(
            min=PAR_RADIUS_MIN,
            max=PAR_RADIUS_MAX,
        )

        for task_index in unique_tasks:
            task_class_mask = (labels == class_index) & (tasks == task_index)
            if task_class_mask.sum().item() < min_count:
                continue

            task_center = F.normalize(
                features[task_class_mask].mean(dim=0),
                dim=0,
                eps=1e-6,
            )
            distance = 1.0 - F.cosine_similarity(
                task_center.unsqueeze(0),
                class_prototype.unsqueeze(0),
                dim=1,
            ).mean()
            overflow = F.relu(distance - class_radius)
            loss = loss + overflow.pow(2)
            valid_center_count += 1

    if valid_center_count > 0:
        loss = loss / valid_center_count
    return loss


class PAR:
    """Prototype Alignment Radius strategy applied to the fused representation z.

    Call ``compute`` before backpropagation and ``update`` after ``optimizer.step()``.
    """

    def __init__(
        self,
        num_classes: int = 3,
        feature_dim: int = 64,
        device: torch.device | str = "cuda",
        momentum: float = PAR_PROTOTYPE_MOMENTUM,
        min_count: int = PAR_MIN_COUNT,
        pull_weight: float = PAR_PULL_WEIGHT,
        task_center_weight: float = PAR_TASK_CENTER_WEIGHT,
        loss_weight: float = PAR_LOSS_WEIGHT,
    ) -> None:
        self.num_classes = num_classes
        self.min_count = min_count
        self.pull_weight = pull_weight
        self.task_center_weight = task_center_weight
        self.loss_weight = loss_weight
        self.bank = GlobalPrototypeBank(
            num_classes=num_classes,
            feature_dim=feature_dim,
            momentum=momentum,
            min_count=min_count,
            device=device,
        )

    def compute(
        self,
        fused_features: Tensor,
        labels: Tensor,
        tasks: Tensor,
    ) -> Dict[str, Tensor]:
        normalized_features = F.normalize(fused_features, dim=1)

        prototypes, valid_mask, radii = build_reference_prototypes(
            normalized_features.detach(),
            labels,
            prototype_bank=self.bank,
            num_classes=self.num_classes,
            min_count=self.min_count,
        )

        pull_loss = sample_radius_alignment_loss(
            normalized_features,
            labels,
            prototypes,
            valid_mask,
            radii,
        )
        task_center_loss = task_center_radius_alignment_loss(
            normalized_features,
            labels,
            tasks,
            prototypes,
            valid_mask,
            radii,
            min_count=self.min_count,
        )
        par_loss = (
            self.pull_weight * pull_loss
            + self.task_center_weight * task_center_loss
        )

        return {
            "pull": pull_loss,
            "task_center": task_center_loss,
            "par": par_loss,
            "weighted_par": self.loss_weight * par_loss,
        }

    @torch.no_grad()
    def update(self, fused_features: Tensor, labels: Tensor) -> None:
        normalized_features = F.normalize(fused_features.detach(), dim=1)
        self.bank.update(normalized_features, labels)


# =============================================================================
# MBPnet
# =============================================================================


class MBPNet(nn.Module):
    """MBPnet backbone (MTE + ChannelAggregation + BRF + classifier).

    PAR is instantiated separately as an auxiliary training loss.
    """

    def __init__(
        self,
        num_classes: int = 3,
        d_model: int = 64,
        patch_size: int = 32,
        stride: int = 16,
    ) -> None:
        super().__init__()

        # One MTE shared by the three EEG channels, and one MTE for ECG.
        self.eeg_mte = MTE(
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            dropout=0.1,
        )
        self.ecg_mte = MTE(
            d_model=d_model,
            patch_size=patch_size,
            stride=stride,
            dropout=0.1,
        )
        self.channel_aggregation = ChannelAggregation(d_model=d_model)
        self.brf = BRF(
            d_model=d_model,
            num_bottlenecks=4,
            num_heads=4,
            dropout=0.1,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        eeg: Tensor,
        ecg: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # EEG: [B, 3, T] or [B, 1, 3, T]; ECG: [B, 1, T] or [B, 1, 1, T]
        if eeg.ndim == 4:
            eeg = eeg.squeeze(1)
        if ecg.ndim == 4:
            ecg = ecg.squeeze(1)

        if eeg.ndim != 3 or eeg.size(1) != 3:
            raise ValueError(
                f"EEG input must have shape [B, 3, T] or [B, 1, 3, T], got {tuple(eeg.shape)}."
            )
        if ecg.ndim != 3 or ecg.size(1) != 1:
            raise ValueError(
                f"ECG input must have shape [B, 1, T] or [B, 1, 1, T], got {tuple(ecg.shape)}."
            )

        eeg_channel_tokens = [
            self.eeg_mte(eeg[:, channel_index : channel_index + 1, :])
            for channel_index in range(eeg.size(1))
        ]
        eeg_tokens = self.channel_aggregation(eeg_channel_tokens)
        ecg_tokens = self.ecg_mte(ecg)

        (
            fused_representation,
            cross_modal_representation,
            eeg_representation,
            ecg_representation,
            _,
            _,
        ) = self.brf(eeg_tokens, ecg_tokens)

        logits = self.classifier(fused_representation)

        return (
            logits,
            fused_representation,
            cross_modal_representation,
            eeg_representation,
            ecg_representation,
        )


__all__ = [
    "MTE",
    "ChannelAggregation",
    "BRF",
    "PAR",
    "MBPNet",
    "GlobalPrototypeBank",
    "build_reference_prototypes",
    "sample_radius_alignment_loss",
    "task_center_radius_alignment_loss",
]