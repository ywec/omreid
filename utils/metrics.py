from prettytable import PrettyTable
import torch
import torch.nn.functional as F
import logging
from typing import List, Tuple, Callable, Dict, Any

def _fmt3(v):
    try:
        return f"{float(v):.3f}"
    except Exception:
        return v

def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
    if get_mAP:
        indices = torch.argsort(similarity, dim=1, descending=True)
    else:
        # acclerate sort with topk
        _, indices = torch.topk(
            similarity, k=max_rank, dim=1, largest=True, sorted=True
        )  # q * topk
    indices = indices.to(g_pids.device)
    pred_labels = g_pids[indices]  # q * k
    matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

    all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
    all_cmc[all_cmc > 1] = 1
    all_cmc = all_cmc.float().mean(0) * 100
    # all_cmc = all_cmc[topk - 1]

    if not get_mAP:
        return all_cmc, torch.tensor(0), torch.tensor(0), indices

    num_rel = matches.sum(1)  # q
    tmp_cmc = matches.cumsum(1)  # q * k

    inp = [tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.) for i, match_row in enumerate(matches)]
    mINP = torch.cat(inp).mean() * 100

    tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
    tmp_cmc = torch.stack(tmp_cmc, 1) * matches
    AP = tmp_cmc.sum(1) / num_rel  # q
    mAP = AP.mean() * 100

    return all_cmc, mAP, mINP, indices


class Evaluator():
    def __init__(self, gallery_loader, get_mAP=True, **query_loaders):
        self.gallery_loader = gallery_loader
        self.query_loaders = query_loaders
        self.get_mAP = get_mAP
        self.logger = logging.getLogger("ORBench.eval")

        # Define modality encoding methods
        self.modality_encoders = {
            'nir': 'encode_nir_cls',
            'cp': 'encode_cp_cls',
            'sk': 'encode_sk_cls',
            'text': 'encode_text_cls'
        }

        self.modality_embed_encoders = {
            'nir': 'encode_nir_embeds',
            'cp': 'encode_cp_embeds',
            'sk': 'encode_sk_embeds',
            'text': 'encode_text_embeds'
        }

    def _extract_single_modality_features(self, model, loader, modality):
        """Extract features for single modality"""
        model = model.eval()
        device = next(model.parameters()).device
        qids, qfeats = [], []

        for pid, img in loader:
            img = img.to(device)
            with torch.no_grad():
                encoder_method = getattr(model, self.modality_encoders[modality])
                img_feat = encoder_method(img)
            qids.append(pid.view(-1))
            qfeats.append(img_feat)

        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        qfeats = F.normalize(qfeats, p=2, dim=1)

        return qids, qfeats

    def _extract_multi_modality_features(self, model, loader, modalities):
        """Extract fused features for multiple modalities"""
        model = model.eval()
        device = next(model.parameters()).device
        qids, qfeats = [], []

        # Determine the number of images based on modalities count
        num_modalities = len(modalities)

        for batch in loader:
            pid = batch[0]
            imgs = [img.to(device) for img in batch[1:1 + num_modalities]]

            with torch.no_grad():
                embeds = []
                for i, modality in enumerate(modalities):
                    encoder_method = getattr(model, self.modality_embed_encoders[modality])

                    if modality == 'text':
                        # Text encoding returns tuple, we take the second element
                        _, text_embed = encoder_method(imgs[i])
                        embeds.append(text_embed)
                    else:
                        embed = encoder_method(imgs[i])
                        embeds.append(embed)

                # Concatenate all embeddings and fuse
                combined = torch.cat(embeds, dim=1)
                fusion_feats = model.mm_fusion(combined, combined, combined)

            qids.append(pid.view(-1))
            qfeats.append(fusion_feats)

        qids = torch.cat(qids, 0)
        qfeats = torch.cat(qfeats, 0)
        qfeats = F.normalize(qfeats, p=2, dim=1)

        return qids, qfeats

    def _evaluate_modality(self, gfeats, gids, model, loader, modalities):
        """Generic evaluation function for any modality combination"""
        if len(modalities) == 1:
            qids, qfeats = self._extract_single_modality_features(model, loader, modalities[0])
        else:
            qids, qfeats = self._extract_multi_modality_features(model, loader, modalities)

        similarity = qfeats @ gfeats.t()
        t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
            similarity=similarity,
            q_pids=qids,
            g_pids=gids,
            max_rank=10,
            get_mAP=self.get_mAP
        )

        t2i_cmc, t2i_mAP, t2i_mINP = t2i_cmc.cpu().numpy(), t2i_mAP.cpu().numpy(), t2i_mINP.cpu().numpy()
        return t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP

    def _get_modality_combinations(self):
        """Define all modality combinations and their corresponding loaders"""
        return [
            # Single modalities
            ('NIR', ['nir'], self.query_loaders.get('nir_query_loader')),
            ('CP', ['cp'], self.query_loaders.get('cp_query_loader')),
            ('SK', ['sk'], self.query_loaders.get('sk_query_loader')),
            ('TEXT', ['text'], self.query_loaders.get('text_query_loader')),

            # Two modalities
            ('NIR+CP', ['nir', 'cp'], self.query_loaders.get('nir_cp_query_loader')),
            ('CP+NIR', ['cp', 'nir'], self.query_loaders.get('cp_nir_query_loader')),
            ('NIR+SK', ['nir', 'sk'], self.query_loaders.get('nir_sk_query_loader')),
            ('SK+NIR', ['sk', 'nir'], self.query_loaders.get('sk_nir_query_loader')),
            ('NIR+TEXT', ['nir', 'text'], self.query_loaders.get('nir_text_query_loader')),
            ('TEXT+NIR', ['text', 'nir'], self.query_loaders.get('text_nir_query_loader')),
            ('CP+SK', ['cp', 'sk'], self.query_loaders.get('cp_sk_query_loader')),
            ('SK+CP', ['sk', 'cp'], self.query_loaders.get('sk_cp_query_loader')),
            ('CP+TEXT', ['cp', 'text'], self.query_loaders.get('cp_text_query_loader')),
            ('TEXT+CP', ['text', 'cp'], self.query_loaders.get('text_cp_query_loader')),
            ('SK+TEXT', ['sk', 'text'], self.query_loaders.get('sk_text_query_loader')),
            ('TEXT+SK', ['text', 'sk'], self.query_loaders.get('text_sk_query_loader')),

            # Three modalities
            ('NIR+CP+SK', ['nir', 'cp', 'sk'], self.query_loaders.get('nir_cp_sk_query_loader')),
            ('CP+NIR+SK', ['cp', 'nir', 'sk'], self.query_loaders.get('cp_nir_sk_query_loader')),
            ('SK+NIR+CP', ['sk', 'nir', 'cp'], self.query_loaders.get('sk_nir_cp_query_loader')),
            ('NIR+CP+TEXT', ['nir', 'cp', 'text'], self.query_loaders.get('nir_cp_text_query_loader')),
            ('CP+NIR+TEXT', ['cp', 'nir', 'text'], self.query_loaders.get('cp_nir_text_query_loader')),
            ('TEXT+NIR+CP', ['text', 'nir', 'cp'], self.query_loaders.get('text_nir_cp_query_loader')),
            ('NIR+SK+TEXT', ['nir', 'sk', 'text'], self.query_loaders.get('nir_sk_text_query_loader')),
            ('SK+NIR+TEXT', ['sk', 'nir', 'text'], self.query_loaders.get('sk_nir_text_query_loader')),
            ('TEXT+NIR+SK', ['text', 'nir', 'sk'], self.query_loaders.get('text_nir_sk_query_loader')),
            ('CP+SK+TEXT', ['cp', 'sk', 'text'], self.query_loaders.get('cp_sk_text_query_loader')),
            ('SK+CP+TEXT', ['sk', 'cp', 'text'], self.query_loaders.get('sk_cp_text_query_loader')),
            ('TEXT+CP+SK', ['text', 'cp', 'sk'], self.query_loaders.get('text_cp_sk_query_loader')),

            # Four modalities
            ('NIR+CP+SK+TEXT', ['nir', 'cp', 'sk', 'text'], self.query_loaders.get('nir_cp_sk_text_query_loader')),
            ('CP+NIR+SK+TEXT', ['cp', 'nir', 'sk', 'text'], self.query_loaders.get('cp_nir_sk_text_query_loader')),
            ('SK+NIR+CP+TEXT', ['sk', 'nir', 'cp', 'text'], self.query_loaders.get('sk_nir_cp_text_query_loader')),
            ('TEXT+NIR+CP+SK', ['text', 'nir', 'cp', 'sk'], self.query_loaders.get('text_nir_cp_sk_query_loader')),
        ]

    def eval(self, model):
        model = model.eval()
        device = next(model.parameters()).device

        # Extract gallery features
        gids, gfeats = [], []
        for pid, img in self.gallery_loader:
            img = img.to(device)
            with torch.no_grad():
                img_feat = model.encode_rgb_cls(img)
            gids.append(pid.view(-1))
            gfeats.append(img_feat)
        gids = torch.cat(gids, 0)
        gfeats = torch.cat(gfeats, 0)
        gfeats = F.normalize(gfeats, p=2, dim=1)

        eval_results = {}
        modality_combinations = self._get_modality_combinations()

        # Group evaluations by modality count for organized printing
        modality_groups = {
            1: "One Modality Evaluating...",
            2: "\nTwo Modalities Evaluating...",
            3: "\nThree Modalities Evaluating...",
            4: "\nFour Modalities Evaluating..."
        }

        current_modality_count = 0

        for task_name, modalities, loader in modality_combinations:
            if loader is None:
                continue

            modality_count = len(modalities)
            if modality_count != current_modality_count:
                current_modality_count = modality_count
                print(modality_groups.get(modality_count, ""))

            result = self._evaluate_modality(gfeats, gids, model, loader, modalities)
            eval_results[task_name] = result

            print(
                f"{task_name}: R1={result[0]:.3f}, R5={result[1]:.3f}, "
                f"R10={result[2]:.3f}, mAP={result[3]:.3f}, mINP={result[4]:.3f}"
            )

        # Build summary table
        table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
        for task_name, _, _ in modality_combinations:
            if task_name in eval_results:
                result = eval_results[task_name]
                table.add_row([task_name, result[0], result[1], result[2], result[3], result[4]])

        # Calculate averages
        one_aver = self.table_average_calculation(table, 0, 4)
        two_aver = self.table_average_calculation(table, 4, 16)
        three_aver = self.table_average_calculation(table, 16, 28)
        four_aver = self.table_average_calculation(table, 28, 32)

        table.add_row(['ONE_AVER', one_aver[0], one_aver[1], one_aver[2], one_aver[3], one_aver[4]])
        table.add_row(['TWO_AVER', two_aver[0], two_aver[1], two_aver[2], two_aver[3], two_aver[4]])
        table.add_row(['THREE_AVER', three_aver[0], three_aver[1], three_aver[2], three_aver[3], three_aver[4]])
        table.add_row(['FOUR_AVER', four_aver[0], four_aver[1], four_aver[2], four_aver[3], four_aver[4]])

        # Format table
        try:
            # 新版：按列定制格式
            for field in ["R1", "R5", "R10", "mAP", "mINP"]:
                table.custom_format[field] = lambda f, v, _f=_fmt3: _f(v)
        except AttributeError:
            # 旧版：没有 custom_format，用全局的小数格式，并把现有行转成字符串
            table.float_format = "0.3"
            table._rows = [[row[0]] + [_fmt3(x) for x in row[1:]] for row in table._rows]

        self.logger.info('\n' + str(table))

        return (one_aver[0] + two_aver[0] + three_aver[0] + four_aver[0]) / 4

    def table_average_calculation(self, table, first, last):
        selected_rows = table._rows[first:last]
        column_sums = [0] * (len(table.field_names) - 1)
        num_rows = len(selected_rows)

        for row in selected_rows:
            for i, value in enumerate(row[1:]):
                column_sums[i] += value

        averages = [sum_val / num_rows for sum_val in column_sums]
        return averages

# import logging
# from prettytable import PrettyTable
# import torch
# import torch.nn.functional as F


# def rank(similarity, q_pids, g_pids, max_rank=10, get_mAP=True):
#     if get_mAP:
#         indices = torch.argsort(similarity, dim=1, descending=True)
#     else:
#         # acclerate sort with topk
#         _, indices = torch.topk(
#             similarity, k=max_rank, dim=1, largest=True, sorted=True
#         )  # q * topk
#     indices = indices.to(g_pids.device)
#     pred_labels = g_pids[indices]  # q * k
#     matches = pred_labels.eq(q_pids.view(-1, 1))  # q * k

#     all_cmc = matches[:, :max_rank].cumsum(1)  # cumulative sum
#     all_cmc[all_cmc > 1] = 1
#     all_cmc = all_cmc.float().mean(0) * 100

#     if not get_mAP:
#         return all_cmc, torch.tensor(0), torch.tensor(0), indices

#     num_rel = matches.sum(1)  # q
#     tmp_cmc = matches.cumsum(1)  # q * k

#     inp = [
#         tmp_cmc[i][match_row.nonzero()[-1]] / (match_row.nonzero()[-1] + 1.0)
#         for i, match_row in enumerate(matches)
#     ]
#     mINP = torch.cat(inp).mean() * 100

#     tmp_cmc = [tmp_cmc[:, i] / (i + 1.0) for i in range(tmp_cmc.shape[1])]
#     tmp_cmc = torch.stack(tmp_cmc, 1) * matches
#     AP = tmp_cmc.sum(1) / num_rel  # q
#     mAP = AP.mean() * 100

#     return all_cmc, mAP, mINP, indices


# class Evaluator:
#     def __init__(self, gallery_loader, get_mAP=True, **query_loaders):
#         self.gallery_loader = gallery_loader
#         self.query_loaders = query_loaders
#         self.get_mAP = get_mAP
#         self.logger = logging.getLogger("ORBench.eval")

#         # 编码函数名称映射，便于在统一的生成-融合流程中动态调用
#         self.modality_embed_encoders = {
#             "nir": "encode_nir_embeds",
#             "cp": "encode_cp_embeds",
#             "sk": "encode_sk_embeds",
#             "text": "encode_text_embeds",
#         }

#     def _get_first_param(self, model, *module_names):
#         """Get the first parameter from available modules for dtype/device hints."""

#         for name in module_names:
#             module = getattr(model, name, None)
#             if module is None:
#                 continue
#             try:
#                 return next(module.parameters())
#             except (AttributeError, TypeError, StopIteration):
#                 continue

#         raise AttributeError(
#             f"None of the modules {module_names} on model '{type(model)}' expose parameters."
#         )

#     def _extract_fused_features(self, model, loader, modalities):
#         """所有模态组合统一先走 mm_generation，再通过 rgb_center 查询进入 mm_fusion。"""

#         model = model.eval()
#         # 使用生成与融合模块自身的 dtype / device，避免混精度下参数与输入不匹配
#         mge_param = self._get_first_param(model, "mm_generation")
#         # build_v1 使用 modal_moe 作为融合模块，build 使用 mm_fusion（nn.Module）
#         mfu_param = self._get_first_param(model, "modal_moe", "mm_fusion")
#         device = mge_param.device
#         mge_dtype = mge_param.dtype
#         mfu_dtype = mfu_param.dtype
#         qids, qfeats = [], []

#         num_modalities = len(modalities)

#         for batch in loader:
#             pid = batch[0]
#             imgs = batch[1 : 1 + num_modalities]

#             # 单模态 loader 直接返回 (pid, img)，此时需要补一层列表
#             if num_modalities == 1:
#                 imgs = [batch[1]]

#             imgs = [img.to(device) for img in imgs]

#             with torch.no_grad():
#                 # 先构造 mm_generation 需要的入参，再按顺序填充
#                 # ========== 单模态：直接用 encoder CLS 特征，不走 MGE / Fusion ==========
#                 if num_modalities == 1:
#                     m = modalities[0]
#                     if m in ("text", "txt"):
#                         feat = model.encode_text_cls(imgs[0])   # [B, D]
#                     else:
#                         # m: nir/cp/sk
#                         feat = getattr(model, f"encode_{m}_cls")(imgs[0])  # [B, D]

#                     qids.append(pid.view(-1))
#                     qfeats.append(feat)
#                     continue
                    
#                 mge_kwargs = {
#                     "nir_patches": None,
#                     "nir_cls": None,
#                     "cp_patches": None,
#                     "cp_cls": None,
#                     "sk_patches": None,
#                     "sk_cls": None,
#                     "txt_patches": None,
#                     "txt_cls": None,
#                 }

#                 for i, modality in enumerate(modalities):
#                     encoder_method = getattr(
#                         model, self.modality_embed_encoders[modality]
#                     )

#                     if modality == "text":
#                         txt_cls, txt_tokens = encoder_method(imgs[i])
#                         mge_kwargs["txt_patches"] = txt_tokens.to(
#                             device=device, dtype=mge_dtype
#                         )
#                         mge_kwargs["txt_cls"] = txt_cls.to(device=device, dtype=mge_dtype)
#                     else:
#                         tokens = encoder_method(imgs[i]).to(device=device, dtype=mge_dtype)
#                         mge_kwargs[f"{modality}_patches"] = tokens[:, 1:, :]
#                         mge_kwargs[f"{modality}_cls"] = tokens[:, 0, :]

#                 # 评测时开启 autocast，避免 MGE 内部出现 Float vs Half（LN/Attention 常见）
#                 if device.type == "cuda" and mge_dtype in (torch.float16, torch.bfloat16):
#                     with torch.autocast(device_type="cuda", dtype=mge_dtype):
#                         mge_outs = model.mm_generation(**mge_kwargs)
#                 else:
#                     mge_outs = model.mm_generation(**mge_kwargs)

#                 # build_v1 规定使用 rgb_center 作为融合查询
#                 batch_size = pid.shape[0] if pid.ndim > 0 else 1
#                 rgb_query = (
#                     model.rgb_center.to(device=device, dtype=mfu_dtype).expand(
#                         batch_size, -1
#                     )
#                 )

#                 fusion_feats = model.mm_fusion(
#                     query=rgb_query,
#                     cls_nir=mge_outs.get("nir"),
#                     cls_cp=mge_outs.get("cp"),
#                     cls_sk=mge_outs.get("sk"),
#                     cls_txt=mge_outs.get("txt"),
#                 )

#             qids.append(pid.view(-1))
#             qfeats.append(fusion_feats)

#         qids = torch.cat(qids, 0)
#         qfeats = torch.cat(qfeats, 0)
#         qfeats = F.normalize(qfeats, p=2, dim=1)

#         return qids, qfeats

#     def _evaluate_modality(self, gfeats, gids, model, loader, modalities):
#         """通用的模态组合评测流程。"""

#         qids, qfeats = self._extract_fused_features(model, loader, modalities)

#         similarity = qfeats @ gfeats.t()
#         t2i_cmc, t2i_mAP, t2i_mINP, _ = rank(
#             similarity=similarity,
#             q_pids=qids,
#             g_pids=gids,
#             max_rank=10,
#             get_mAP=self.get_mAP,
#         )

#         t2i_cmc = t2i_cmc.cpu().numpy()
#         t2i_mAP = t2i_mAP.cpu().numpy()
#         t2i_mINP = t2i_mINP.cpu().numpy()

#         return t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_mAP, t2i_mINP

#     def _get_modality_combinations(self):
#         """构建所有需要评测的模态组合及对应 dataloader。"""

#         return [
#             # 单模态
#             ("NIR", ["nir"], self.query_loaders.get("nir_query_loader")),
#             ("CP", ["cp"], self.query_loaders.get("cp_query_loader")),
#             ("SK", ["sk"], self.query_loaders.get("sk_query_loader")),
#             ("TEXT", ["text"], self.query_loaders.get("text_query_loader")),

#             # 双模态
#             ("NIR+CP", ["nir", "cp"], self.query_loaders.get("nir_cp_query_loader")),
#             ("CP+NIR", ["cp", "nir"], self.query_loaders.get("cp_nir_query_loader")),
#             ("NIR+SK", ["nir", "sk"], self.query_loaders.get("nir_sk_query_loader")),
#             ("SK+NIR", ["sk", "nir"], self.query_loaders.get("sk_nir_query_loader")),
#             ("NIR+TEXT", ["nir", "text"], self.query_loaders.get("nir_text_query_loader")),
#             ("TEXT+NIR", ["text", "nir"], self.query_loaders.get("text_nir_query_loader")),
#             ("CP+SK", ["cp", "sk"], self.query_loaders.get("cp_sk_query_loader")),
#             ("SK+CP", ["sk", "cp"], self.query_loaders.get("sk_cp_query_loader")),
#             ("CP+TEXT", ["cp", "text"], self.query_loaders.get("cp_text_query_loader")),
#             ("TEXT+CP", ["text", "cp"], self.query_loaders.get("text_cp_query_loader")),
#             ("SK+TEXT", ["sk", "text"], self.query_loaders.get("sk_text_query_loader")),
#             ("TEXT+SK", ["text", "sk"], self.query_loaders.get("text_sk_query_loader")),

#             # 三模态
#             (
#                 "NIR+CP+SK",
#                 ["nir", "cp", "sk"],
#                 self.query_loaders.get("nir_cp_sk_query_loader"),
#             ),
#             (
#                 "CP+NIR+SK",
#                 ["cp", "nir", "sk"],
#                 self.query_loaders.get("cp_nir_sk_query_loader"),
#             ),
#             (
#                 "SK+NIR+CP",
#                 ["sk", "nir", "cp"],
#                 self.query_loaders.get("sk_nir_cp_query_loader"),
#             ),
#             (
#                 "NIR+CP+TEXT",
#                 ["nir", "cp", "text"],
#                 self.query_loaders.get("nir_cp_text_query_loader"),
#             ),
#             (
#                 "CP+NIR+TEXT",
#                 ["cp", "nir", "text"],
#                 self.query_loaders.get("cp_nir_text_query_loader"),
#             ),
#             (
#                 "TEXT+NIR+CP",
#                 ["text", "nir", "cp"],
#                 self.query_loaders.get("text_nir_cp_query_loader"),
#             ),
#             (
#                 "NIR+SK+TEXT",
#                 ["nir", "sk", "text"],
#                 self.query_loaders.get("nir_sk_text_query_loader"),
#             ),
#             (
#                 "SK+NIR+TEXT",
#                 ["sk", "nir", "text"],
#                 self.query_loaders.get("sk_nir_text_query_loader"),
#             ),
#             (
#                 "TEXT+NIR+SK",
#                 ["text", "nir", "sk"],
#                 self.query_loaders.get("text_nir_sk_query_loader"),
#             ),
#             (
#                 "CP+SK+TEXT",
#                 ["cp", "sk", "text"],
#                 self.query_loaders.get("cp_sk_text_query_loader"),
#             ),
#             (
#                 "SK+CP+TEXT",
#                 ["sk", "cp", "text"],
#                 self.query_loaders.get("sk_cp_text_query_loader"),
#             ),
#             (
#                 "TEXT+CP+SK",
#                 ["text", "cp", "sk"],
#                 self.query_loaders.get("text_cp_sk_query_loader"),
#             ),

#             # 四模态
#             (
#                 "NIR+CP+SK+TEXT",
#                 ["nir", "cp", "sk", "text"],
#                 self.query_loaders.get("nir_cp_sk_text_query_loader"),
#             ),
#             (
#                 "CP+NIR+SK+TEXT",
#                 ["cp", "nir", "sk", "text"],
#                 self.query_loaders.get("cp_nir_sk_text_query_loader"),
#             ),
#             (
#                 "SK+NIR+CP+TEXT",
#                 ["sk", "nir", "cp", "text"],
#                 self.query_loaders.get("sk_nir_cp_text_query_loader"),
#             ),
#             (
#                 "TEXT+NIR+CP+SK",
#                 ["text", "nir", "cp", "sk"],
#                 self.query_loaders.get("text_nir_cp_sk_query_loader"),
#             ),
#         ]

#     def _init_group_stats(self):
#         return {
#             1: {"sum": [0.0] * 5, "count": 0},
#             2: {"sum": [0.0] * 5, "count": 0},
#             3: {"sum": [0.0] * 5, "count": 0},
#             4: {"sum": [0.0] * 5, "count": 0},
#         }

#     def _update_group_stats(self, stats, modality_count, result):
#         stats_group = stats[modality_count]
#         stats_group["sum"] = [a + b for a, b in zip(stats_group["sum"], result)]
#         stats_group["count"] += 1

#     def _finalize_group_average(self, stats_group):
#         if stats_group["count"] == 0:
#             return [0.0] * 5
#         return [v / stats_group["count"] for v in stats_group["sum"]]

#     def eval(self, model):
#         model = model.eval()
#         fusion_param = self._get_first_param(model, "modal_moe", "mm_fusion")
#         device = fusion_param.device
#         fusion_dtype = fusion_param.dtype

#         # 1) 提前抽取 gallery 特征，后续检索直接矩阵相似度
#         gids, gfeats = [], []
#         for pid, img in self.gallery_loader:
#             img = img.to(device)
#             with torch.no_grad():
#                 img_feat = model.encode_rgb_cls(img)
#                 img_feat = img_feat.to(device=device, dtype=fusion_dtype)
#             gids.append(pid.view(-1))
#             gfeats.append(img_feat)
#         gids = torch.cat(gids, 0)
#         gfeats = torch.cat(gfeats, 0)
#         gfeats = F.normalize(gfeats, p=2, dim=1)

#         # 2) 针对所有模态组合做评测
#         eval_results = {}
#         modality_combinations = self._get_modality_combinations()
#         modality_groups = {
#             1: "One Modality Evaluating...",
#             2: "\nTwo Modalities Evaluating...",
#             3: "\nThree Modalities Evaluating...",
#             4: "\nFour Modalities Evaluating...",
#         }

#         current_modality_count = 0
#         group_stats = self._init_group_stats()

#         for task_name, modalities, loader in modality_combinations:
#             if loader is None:
#                 continue

#             modality_count = len(modalities)
#             if modality_count != current_modality_count:
#                 current_modality_count = modality_count
#                 print(modality_groups.get(modality_count, ""))

#             result = self._evaluate_modality(gfeats, gids, model, loader, modalities)
#             eval_results[task_name] = result

#             self._update_group_stats(group_stats, modality_count, result)

#             print(
#                 f"{task_name}: R1={result[0]:.3f}, R5={result[1]:.3f}, "
#                 f"R10={result[2]:.3f}, mAP={result[3]:.3f}, mINP={result[4]:.3f}"
#             )

#         # 3) 汇总表格
#         table = PrettyTable(["task", "R1", "R5", "R10", "mAP", "mINP"])
#         for task_name, _, _ in modality_combinations:
#             if task_name in eval_results:
#                 result = eval_results[task_name]
#                 table.add_row(
#                     [task_name, result[0], result[1], result[2], result[3], result[4]]
#                 )

#         one_aver = self._finalize_group_average(group_stats[1])
#         two_aver = self._finalize_group_average(group_stats[2])
#         three_aver = self._finalize_group_average(group_stats[3])
#         four_aver = self._finalize_group_average(group_stats[4])

#         table.add_row(["ONE_AVER", *one_aver])
#         table.add_row(["TWO_AVER", *two_aver])
#         table.add_row(["THREE_AVER", *three_aver])
#         table.add_row(["FOUR_AVER", *four_aver])

#         def _fmt3(v):
#             # 兼容 torch tensor / numpy / python float
#             if torch.is_tensor(v):
#                 v = v.item()
#             return f"{float(v):.3f}"

#         try:
#             for field in ["R1", "R5", "R10", "mAP", "mINP"]:
#                 table.custom_format[field] = lambda f, v, _f=_fmt3: _f(v)
#         except AttributeError:
#             table._rows = [[row[0]] + [_fmt3(x) for x in row[1:]] for row in table._rows]


#         self.logger.info("\n" + str(table))

#         return (one_aver[0] + two_aver[0] + three_aver[0] + four_aver[0]) / 4
