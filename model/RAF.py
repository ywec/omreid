import torch
import torch.nn as nn
from typing import Tuple


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Expert(nn.Module):
    def __init__(self, input_dim: int):
        super(Expert, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            QuickGELU(),
            nn.BatchNorm1d(input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ExpertHead(nn.Module):
    def __init__(self, head_dim: int, num_experts: int):
        super(ExpertHead, self).__init__()
        self.experts = nn.ModuleList([Expert(head_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(head_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, head_dim]
        B, D = x.shape

        gate_logits = self.gate(x)             # [B, num_experts]
        weights = gate_logits.softmax(dim=-1)  # [B, num_experts]

        # 所有 expert 的输出：[B, D, num_experts]
        expert_outs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # [B, D, E]

        # 按 expert 维度加权求和:  y[b, d] = sum_e outs[b,d,e] * w[b,e]
        y = torch.einsum("bde,be->bd", expert_outs, weights)  # [B, D]
        return y


class MoM(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, head: int):
        super(MoM, self).__init__()
        assert input_dim % head == 0, "input_dim 必须能被 head 整除"

        self.input_dim = input_dim
        self.head = head
        self.head_dim = input_dim // head

        self.heads = nn.ModuleList(
            [ExpertHead(self.head_dim, num_experts) for _ in range(head)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D] 或 [D]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # [1, D]

        B, D = x.shape
        assert D == self.input_dim, f"输入维度 {D} 与初始化的 input_dim {self.input_dim} 不一致"

        # 按 head 切块，每块 [B, head_dim]
        chunks = torch.chunk(x, self.head, dim=-1)
        out_chunks = []

        for h, head_moe in enumerate(self.heads):
            out_h = head_moe(chunks[h])   # [B, head_dim]
            out_chunks.append(out_h)

        # 拼回 [B, D]
        out = torch.cat(out_chunks, dim=-1)
        return out


class ModalMoEFusion(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, head: int):
        super(ModalMoEFusion, self).__init__()
        self.input_dim = input_dim
        self.moe = MoM(input_dim=input_dim, num_experts=num_experts, head=head)
        self.scale = input_dim ** -0.5

    def _ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        """把 [D] 变成 [1, D]，方便统一处理。"""
        if x is None:
            return None
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def forward(
        self,
        query: torch.Tensor,
        cls_nir: torch.Tensor,
        cls_cp: torch.Tensor,
        cls_sk: torch.Tensor,
        cls_txt: torch.Tensor,
    ):

        # 统一成 [B, D]
        query   = self._ensure_2d(query)
        cls_nir = self._ensure_2d(cls_nir)
        cls_cp  = self._ensure_2d(cls_cp)
        cls_sk  = self._ensure_2d(cls_sk)
        cls_txt = self._ensure_2d(cls_txt)

        B, D = query.shape

        # 1) 4 个 CLS 过 MoE，得到 4 个增强特征
        feat_nir = self.moe(cls_nir)   # [B, D]
        feat_cp  = self.moe(cls_cp)    # [B, D]
        feat_sk  = self.moe(cls_sk)    # [B, D]
        feat_txt = self.moe(cls_txt)   # [B, D]

        # 2) 用 query 对 4 个 CLS 做 attention 得到权重
        cls_stack = torch.stack([cls_nir, cls_cp, cls_sk, cls_txt], dim=1)  # [B, 4, D]
        # 点积 + 缩放
        scores = (cls_stack * query.unsqueeze(1)).sum(dim=-1) * self.scale  # [B, 4]
        attn_weights = scores.softmax(dim=-1)                               # [B, 4]

        # 3) 按权重对 MoE 输出加权求和
        moe_stack = torch.stack([feat_nir, feat_cp, feat_sk, feat_txt], dim=1)  # [B, 4, D]
        fused_feat = (attn_weights.unsqueeze(-1) * moe_stack).sum(dim=1)       # [B, D]

        return fused_feat
