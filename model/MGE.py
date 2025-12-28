import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Optional, Dict


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class SimpleNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(SimpleNet, self).__init__()
        hidden_dim = in_dim // 2

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            QuickGELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=False,  # 使用 [L, B, D] 格式
        )
        self.ln_1 = nn.LayerNorm(d_model)

        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc",   nn.Linear(d_model, mlp_hidden_dim)),
            ("gelu",   QuickGELU()),
            ("c_proj", nn.Linear(mlp_hidden_dim, d_model)),
        ]))
        self.ln_2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(
            self.ln_1(x),  # Q
            self.ln_1(x),  # K
            self.ln_1(x),  # V
            need_weights=False,
        )[0]

        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPStyleSharedExtractor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.width = embed_dim
        self.layers = num_layers

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                d_model=embed_dim,
                n_head=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)

        # 做一个类似 CLIP 的初始化（简单版）
        scale = self.width ** -0.5
        for block in self.resblocks:
            # attention 部分
            nn.init.normal_(block.attn.in_proj_weight, std=scale)
            if block.attn.in_proj_bias is not None:
                nn.init.zeros_(block.attn.in_proj_bias)
            nn.init.normal_(block.attn.out_proj.weight, std=scale)
            if block.attn.out_proj.bias is not None:
                nn.init.zeros_(block.attn.out_proj.bias)

            # MLP 部分
            nn.init.normal_(block.mlp.c_fc.weight, std=scale)
            if block.mlp.c_fc.bias is not None:
                nn.init.zeros_(block.mlp.c_fc.bias)
            nn.init.normal_(block.mlp.c_proj.weight, std=scale)
            if block.mlp.c_proj.bias is not None:
                nn.init.zeros_(block.mlp.c_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 转成 [L, B, D] 以匹配 MultiheadAttention
        x = x.permute(1, 0, 2)  # [L, B, D]
        for blk in self.resblocks:
            x = blk(x)
        x = self.ln_final(x)
        x = x.permute(1, 0, 2)  # 回到 [B, L, D]
        return x


def build_shared_extractor(
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        dropout: float = 0.0,
    ) -> nn.Module:
    shared_extractor = CLIPStyleSharedExtractor(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        mlp_ratio=4.0,
        dropout=dropout,
    )
    return shared_extractor



class MMGenerationEnhancement(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_query_layers: int = 3,
        mlp_hidden_multiplier: float = 2.0,  # 现在没用到，保留参数不影响
        dropout: float = 0.0,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_query_layers = num_query_layers

        self.shared_extractor = build_shared_extractor(
                embed_dim=embed_dim,
                num_layers=num_query_layers,
                num_heads=num_heads,
                dropout=dropout,
            )

        # -----------------------------
        # 1) 四个模态的 query token [1, 1, D]
        # -----------------------------
        self.nir_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.cp_query  = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.sk_query  = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.txt_query = nn.Parameter(torch.randn(1, 1, embed_dim))

        self._query_dict = {
            "nir": "nir_query",
            "cp":  "cp_query",
            "sk":  "sk_query",
            "txt": "txt_query",
        }

        # -----------------------------
        # 2) 共享的多层 cross-attention（所有模态共享参数）
        # -----------------------------
        self.query_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,  # [B, L, D]
            )
            for _ in range(num_query_layers)
        ])
        self.query_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_query_layers)
        ])

        # -----------------------------
        # 3) 每个模态单独 MLP：输入 [B, 2*D]，输出 [B, D]
        # -----------------------------
        in_dim = 2 * embed_dim
        out_dim = embed_dim

        self.nir_mlp = SimpleNet(in_dim, out_dim)
        self.cp_mlp  = SimpleNet(in_dim, out_dim)
        self.sk_mlp  = SimpleNet(in_dim, out_dim)
        self.txt_mlp = SimpleNet(in_dim, out_dim)

        self._mlp_dict = {
            "nir": "nir_mlp",
            "cp":  "cp_mlp",
            "sk":  "sk_mlp",
            "txt": "txt_mlp",
        }

    # ---------- 内部：多层 query cross-attention ----------
    def _apply_query_attention(
        self,
        query: torch.Tensor,        # [B, 1, D]
        shared_tokens: torch.Tensor # [B, L_total, D]
    ) -> torch.Tensor:
        """
        q_{t+1} = LN(q_t + Attn(q_t, shared_tokens))
        """
        x = query
        for attn, norm in zip(self.query_attn_layers, self.query_norms):
            attn_out, _ = attn(
                x,              # query: [B, 1, D]
                shared_tokens,  # key:   [B, L, D]
                shared_tokens,  # value: [B, L, D]
                need_weights=False,
            )
            x = norm(x + attn_out)
        return x  # [B, 1, D]

    # ------------------------ 前向 ------------------------
    def forward(
        self,
        nir_patches: Optional[torch.Tensor] = None,  # [B, L_n, D] or None
        nir_cls:     Optional[torch.Tensor] = None,  # [B, D] or None
        cp_patches:  Optional[torch.Tensor] = None,  # [B, L_c, D] or None
        cp_cls:      Optional[torch.Tensor] = None,  # [B, D] or None
        sk_patches:  Optional[torch.Tensor] = None,  # [B, L_s, D] or None
        sk_cls:      Optional[torch.Tensor] = None,  # [B, D] or None
        txt_patches: Optional[torch.Tensor] = None,  # [B, L_t, D] or None
        txt_cls:     Optional[torch.Tensor] = None,  # [B, D] or None (可用 eot 当 cls)
    ) -> Dict[str, torch.Tensor]:

        modality_names = ["nir", "cp", "sk", "txt"]
        patch_inputs = {
            "nir": nir_patches,
            "cp":  cp_patches,
            "sk":  sk_patches,
            "txt": txt_patches,
        }
        cls_inputs = {
            "nir": nir_cls,
            "cp":  cp_cls,
            "sk":  sk_cls,
            "txt": txt_cls,
        }

        # 1) 找到第一个真实模态，确定 B / D / device / dtype
        first_valid = None
        for name in modality_names:
            if patch_inputs[name] is not None:
                first_valid = patch_inputs[name]
                break
        if first_valid is None:
            raise ValueError("At least one modality (nir/cp/sk/txt) must be real.")

        B, _, D = first_valid.shape
        device = first_valid.device
        dtype = first_valid.dtype

        # 2) 所有真实模态 patch 拼接 → 共享 Transformer
        real_patches = []
        for name in modality_names:
            p = patch_inputs[name]
            if p is not None:
                real_patches.append(p.to(device=device, dtype=dtype))

        shared_input = torch.cat(real_patches, dim=1)          # [B, L_total, D]
        shared_tokens = self.shared_extractor(shared_input)    # [B, L_total, D]

        # 3) 每个模态用自己的 query 从 shared_tokens 里抽取 new_token
        outputs: Dict[str, torch.Tensor] = {}

        for name in modality_names:
            query_param: nn.Parameter = getattr(self, self._query_dict[name])
            mlp: nn.Module = getattr(self, self._mlp_dict[name])

            # [1, 1, D] -> [B, 1, D]
            q = query_param.to(device=device, dtype=dtype).expand(B, 1, D)

            q_feat = self._apply_query_attention(q, shared_tokens)  # [B, 1, D]
            q_feat = q_feat.squeeze(1)  # [B, D]  新 token

            if patch_inputs[name] is not None and cls_inputs[name] is not None:
                # 真实模态：CLS + new_token -> MLP
                cls = cls_inputs[name].to(device=device, dtype=dtype)  # [B, D]
                fused = torch.cat([cls, q_feat], dim=-1)               # [B, 2D]
                out = mlp(fused)                                       # [B, D]
            else:
                # 缺失模态：直接用 new_token
                out = q_feat

            outputs[name] = out

        return outputs
