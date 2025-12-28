from model import objectives
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights, Transformer, LayerNorm ,QuickGELU
import torch
import copy
import itertools
import torch.nn as nn
from collections import OrderedDict
from .mmencoder_withlora import MMTransformer_withlora
from .MGE import MMGenerationEnhancement, build_shared_extractor
from .RAF import ModalMoEFusion

######################ReID5o Model########################
class VisionTokenizer(nn.Module):
    def __init__(self, conv1, class_embedding, positional_embedding,ln_pre):
        super(VisionTokenizer, self).__init__()
        self.conv1 = conv1
        self.class_embedding = class_embedding
        self.positional_embedding = positional_embedding
        self.ln_pre = ln_pre

    def forward(self, x):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        return x


class MultimodalVisionEncoder(nn.Module):
    def __init__(self, transformer, ln_post, proj):
        super(MultimodalVisionEncoder, self).__init__()
        self.transformer = transformer
        self.ln_post = ln_post
        self.proj = proj

    def forward(self, x,modality='RGB'):
        x = x.permute(1, 0, 2)
        x = self.transformer(x,modality)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)
        x = x @ self.proj
        return x


class CLIPTextEncoder(nn.Module):
    def __init__(self, token_embedding, positional_embedding,transformer,ln_final,text_projection):
        super(CLIPTextEncoder, self).__init__()
        self.token_embedding = token_embedding
        self.positional_embedding = positional_embedding
        self.transformer = transformer
        self.ln_final = ln_final
        self.text_projection = text_projection

    def forward(self, text, dtype):
        x = self.token_embedding(text).type(dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x @ self.text_projection
        eot_embed = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        embeds = x
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return eot_embed, embeds



class ReID5oModel(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()
        # base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size,args.stride_size)
        base_model, base_cfg = build_CLIP_from_openai_pretrained(
            name="/15874700167/pretrain_model/ViT-B-16.pt",
            image_size=args.img_size,
            stride_size=args.stride_size,
        )
        self.embed_dim = base_cfg['embed_dim']  # 512
        self.width = base_cfg['vision_width']  # 768
        self.image_resolution = base_cfg['image_resolution']
        self.encoder_layers = base_cfg['vision_layers']
        self.heads = self.embed_dim //64

        self.mmt_depth = args.mmt_depth

        # 初始化 rgb_center（和 rgb cls 同维度）
        self.rgb_center_momentum = 0.9
        self.register_buffer("rgb_center", torch.zeros(1, self.embed_dim))
        self.register_buffer("rgb_center_initialized", torch.tensor(0, dtype=torch.uint8))


        self.clip_text_encoder = self.build_clip_pretrained_text_encoder(base_model)
        self.rgb_tokenizer = self.build_vision_tokenizer(base_model)
        self.nir_tokenizer = self.build_vision_tokenizer(base_model)
        self.cp_tokenizer = self.build_vision_tokenizer(base_model)
        self.sk_tokenizer = self.build_vision_tokenizer(base_model)
        self.vision_encoder = self.build_vision_encoder(base_model,args)

        self.logit_scale = torch.ones([]) * (1 / args.temperature)


        if 'mm' in args.loss_names:
            self.create_mm_fusion_module()

        if 'id' in args.loss_names:
            self.create_id_classifier()

    @torch.no_grad()
    def update_rgb_center(self, rgb_cls: torch.Tensor):
        batch_center = rgb_cls.mean(dim=0, keepdim=True)

        if self.rgb_center_initialized.item() == 0:
            self.rgb_center.copy_(batch_center)
            self.rgb_center_initialized.fill_(1)
        else:
            m = float(self.rgb_center_momentum)
            self.rgb_center.mul_(m).add_((1.0 - m) * batch_center)

    def create_id_classifier(self):
        print('num_classes:{}'.format(self.num_classes))
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        nn.init.normal_(self.classifier.weight.data, std=0.001)
        nn.init.constant_(self.classifier.bias.data, val=0.0)
    
    def create_mm_fusion_module(self):
        # 1) 先构建共享特征提取器（CLIP-style Transformer）
        shared_extractor = build_shared_extractor(
            embed_dim=self.embed_dim,
            num_layers=self.mmt_depth,
            num_heads=self.embed_dim // 64,
            dropout=0.0,
        )

        # 2) 特征生成 / 增强模块（MGE）
        self.mm_generation = MMGenerationEnhancement(
            embed_dim=self.embed_dim,
            shared_extractor=shared_extractor,
            num_heads=self.embed_dim // 64,
            num_query_layers=self.mmt_depth,
        )

        # 3) 特征混合模块（MoE / RAF）
        moe_num_experts = getattr(self.args, "moe_num_experts", 4)
        moe_head = getattr(self.args, "moe_head", self.heads)

        self.modal_moe = ModalMoEFusion(
            input_dim=self.embed_dim,
            num_experts=moe_num_experts,
            head=moe_head,
        )

    def mm_fusion(
        self,
        query: torch.Tensor,
        cls_nir: torch.Tensor,
        cls_cp: torch.Tensor,
        cls_sk: torch.Tensor,
        cls_txt: torch.Tensor,
    ) -> torch.Tensor:

        fused_feat = self.modal_moe(
            query=query,
            cls_nir=cls_nir,
            cls_cp=cls_cp,
            cls_sk=cls_sk,
            cls_txt=cls_txt,
        )
        return fused_feat

    @property
    def dtype(self):
        return self.rgb_tokenizer.conv1.weight.dtype

    def build_vision_tokenizer(self,base_model):
        conv1 = copy.deepcopy(base_model.visual.conv1)
        cls = copy.deepcopy(base_model.visual.class_embedding)
        pe = copy.deepcopy(base_model.visual.positional_embedding)
        ln_pre = copy.deepcopy(base_model.visual.ln_pre)
        return VisionTokenizer(conv1, cls, pe, ln_pre)

    def build_vision_encoder(self,base_model,args):
        if args.add_lora:
            transformer = MMTransformer_withlora(width=self.width,layers=self.encoder_layers,heads=self.heads,lora_r=args.lora_r, num_loras=args.num_loras,lora_layers=args.lora_layers,lora_mode=args.lora_mode)
            stat = copy.deepcopy(base_model.visual.transformer).state_dict()
            transformer.load_state_dict(stat,strict=False)
            print('Pretrained Multimodal Encoder with LoRAs Loaded, with LoRA_r={}, LoRA_layers={}'.format(args.lora_r,args.lora_layers))
        else:
            transformer = copy.deepcopy(base_model.visual.transformer)
        ln_post = copy.deepcopy(base_model.visual.ln_post)
        proj = copy.deepcopy(base_model.visual.proj)
        encoder = MultimodalVisionEncoder(transformer,ln_post,proj)
        return encoder

    def build_clip_pretrained_text_encoder(self,base_model):
        transformer = copy.deepcopy(base_model.transformer)
        token_embedding = copy.deepcopy(base_model.token_embedding)
        positional_embedding = copy.deepcopy(base_model.positional_embedding)
        ln_final = copy.deepcopy(base_model.ln_final)
        text_projection = copy.deepcopy(base_model.text_projection)
        return CLIPTextEncoder(token_embedding, positional_embedding,transformer,ln_final,text_projection)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def encode_rgb_cls(self,x):
        x = self.rgb_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'RGB')
        return x[:, 0, :].float()

    def encode_nir_cls(self,x):
        x = self.nir_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'NIR')
        return x[:, 0, :].float()

    def encode_cp_cls(self,x):
        x = self.cp_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'CP')
        return x[:, 0, :].float()

    def encode_sk_cls(self,x):
        x = self.sk_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'SK')
        return x[:, 0, :].float()

    def encode_text_cls(self,x):
        x,_ = self.clip_text_encoder(x,self.dtype)
        x = x.float()
        return x

    def encode_rgb_embeds(self,x):
        x = self.rgb_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'RGB')
        return x

    def encode_nir_embeds(self,x):
        x = self.nir_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'NIR')
        return x

    def encode_cp_embeds(self,x):
        x = self.cp_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'CP')
        return x

    def encode_sk_embeds(self,x):
        x = self.sk_tokenizer(x.type(self.dtype))
        x = self.vision_encoder(x,'SK')
        return x

    def encode_text_embeds(self,x):
        eot,x = self.clip_text_encoder(x,self.dtype)
        return eot,x


    def router_multimodal_embeds(self, rgb, nir, cp, sk, text):
        rgb_tokens = self.encode_rgb_embeds(rgb)
        nir_tokens = self.encode_nir_embeds(nir)
        cp_tokens  = self.encode_cp_embeds(cp)
        sk_tokens  = self.encode_sk_embeds(sk)

        text_eot, text_tokens = self.encode_text_embeds(text)

        rgb_tokens  = rgb_tokens.float()
        nir_tokens  = nir_tokens.float()
        cp_tokens   = cp_tokens.float()
        sk_tokens   = sk_tokens.float()
        text_tokens = text_tokens.float()

        combined_embeds_for_one = [
            rgb_tokens,
            nir_tokens,
            cp_tokens,
            sk_tokens,
            text_tokens,
        ]

        mm_embeds = [nir_tokens, cp_tokens, sk_tokens, text_tokens]

        # 2-modality
        combined_embeds_for_two = []
        for combo in itertools.combinations(mm_embeds, 2):
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_two.append(combined)

        # 3-modality
        combined_embeds_for_three = []
        for combo in itertools.combinations(mm_embeds, 3):
            combined = torch.cat(combo, dim=1)
            combined_embeds_for_three.append(combined)

        # 4-modality
        combined_embeds_for_four = [torch.cat(mm_embeds, dim=1)]

        return (
            combined_embeds_for_one,
            combined_embeds_for_two,
            combined_embeds_for_three,
            combined_embeds_for_four,
        )

    def forward(self, batch):
        ret = dict()
        rgbs = batch['rgbs']
        nirs = batch['nirs']
        cps  = batch['cps']
        sks  = batch['sks']
        texts = batch['caption_ids']

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        # ===== 统一算一次 RGB CLS，并更新 center =====
        rgb_cls = self.encode_rgb_cls(rgbs)                # [B, D]
        rgb_center_before = self.rgb_center.clone().detach()
        rgb_query = rgb_center_before.expand(rgb_cls.size(0), -1)   # [B, D]

        # ================== mm_sdm 分支 ==================
        if 'mm_sdm' in self.current_task:
            nir_tokens = self.encode_nir_embeds(nirs)
            cp_tokens  = self.encode_cp_embeds(cps)
            sk_tokens  = self.encode_sk_embeds(sks)
            txt_eot, txt_tokens = self.encode_text_embeds(texts)

            nir_tokens = nir_tokens.float()
            cp_tokens  = cp_tokens.float()
            sk_tokens  = sk_tokens.float()
            txt_tokens = txt_tokens.float()
            txt_eot    = txt_eot.float()

            nir_cls, nir_patches = nir_tokens[:, 0, :], nir_tokens[:, 1:, :]
            cp_cls,  cp_patches  = cp_tokens[:,  0, :], cp_tokens[:,  1:, :]
            sk_cls,  sk_patches  = sk_tokens[:,  0, :], sk_tokens[:,  1:, :]
            txt_cls, txt_patches = txt_eot, txt_tokens

            rgb_feat = rgb_cls   # ★ 修正：原来写成 rgb_feat = rgb_cls(不存在)

            modalities = ["nir", "cp", "sk", "txt"]

            def run_mge_for(mods):
                nir_p = nir_patches if "nir" in mods else None
                nir_c = nir_cls     if "nir" in mods else None

                cp_p  = cp_patches  if "cp"  in mods else None
                cp_c  = cp_cls      if "cp"  in mods else None

                sk_p  = sk_patches  if "sk"  in mods else None
                sk_c  = sk_cls      if "sk"  in mods else None

                txt_p = txt_patches if "txt" in mods else None
                txt_c = txt_cls     if "txt" in mods else None

                mge_outs = self.mm_generation(
                    nir_patches=nir_p, nir_cls=nir_c,
                    cp_patches=cp_p,   cp_cls=cp_c,
                    sk_patches=sk_p,   sk_cls=sk_c,
                    txt_patches=txt_p, txt_cls=txt_c,
                )
                return mge_outs  # {"nir": [B,D], "cp": [B,D], "sk": [B,D], "txt": [B,D]}

            single_sets = [[m] for m in modalities]
            double_sets = [list(c) for c in itertools.combinations(modalities, 2)]
            triple_sets = [list(c) for c in itertools.combinations(modalities, 3)]
            quad_sets   = [modalities]

            cone_feats,  cone_losses  = [], []
            ctwo_feats,  ctwo_losses  = [], []
            cthree_feats, cthree_losses = [], []
            cfour_feats, cfour_losses = [], []

            # ---- 单模态组合 ----
            for mods in single_sets:
                mge_outs = run_mge_for(mods)
                fused = self.mm_fusion(
                    query=rgb_query,
                    cls_nir=mge_outs['nir'],
                    cls_cp=mge_outs['cp'],
                    cls_sk=mge_outs['sk'],
                    cls_txt=mge_outs['txt'],
                )
                cone_feats.append(fused)
                loss = objectives.compute_sdm(rgb_feat, fused, batch['pids'], logit_scale)
                cone_losses.append(loss)

            ret['cone_mmsdm_loss'] = torch.mean(torch.stack(cone_losses))

            # ---- 双模态组合 ----
            for mods in double_sets:
                mge_outs = run_mge_for(mods)
                fused = self.mm_fusion(
                    query=rgb_query,
                    cls_nir=mge_outs['nir'],
                    cls_cp=mge_outs['cp'],
                    cls_sk=mge_outs['sk'],
                    cls_txt=mge_outs['txt'],
                )
                ctwo_feats.append(fused)
                loss = objectives.compute_sdm(rgb_feat, fused, batch['pids'], logit_scale)
                ctwo_losses.append(loss)

            ret['ctwo_mmsdm_loss'] = torch.mean(torch.stack(ctwo_losses))

            # ---- 三模态组合 ----
            for mods in triple_sets:
                mge_outs = run_mge_for(mods)
                fused = self.mm_fusion(
                    query=rgb_query,
                    cls_nir=mge_outs['nir'],
                    cls_cp=mge_outs['cp'],
                    cls_sk=mge_outs['sk'],
                    cls_txt=mge_outs['txt'],
                )
                cthree_feats.append(fused)
                loss = objectives.compute_sdm(rgb_feat, fused, batch['pids'], logit_scale)
                cthree_losses.append(loss)

            ret['cthree_mmsdm_loss'] = torch.mean(torch.stack(cthree_losses))

            # ---- 四模态组合 ----
            for mods in quad_sets:  # 只一个
                mge_outs = run_mge_for(mods)
                fused = self.mm_fusion(
                    query=rgb_query,
                    cls_nir=mge_outs['nir'],
                    cls_cp=mge_outs['cp'],
                    cls_sk=mge_outs['sk'],
                    cls_txt=mge_outs['txt'],
                )
                cfour_feats.append(fused)
                loss = objectives.compute_sdm(rgb_feat, fused, batch['pids'], logit_scale)
                cfour_losses.append(loss)

            ret['cfour_mmsdm_loss'] = torch.mean(torch.stack(cfour_losses))

            # ---- ID loss：仍然 16 个特征 ----
            if 'id' in self.current_task:
                all_feats = (
                    [rgb_feat] +
                    cone_feats +
                    ctwo_feats +
                    cthree_feats +
                    cfour_feats
                )
                assert len(all_feats) == 16

                logits_list = [
                    self.classifier(feat.half()).float()
                    for feat in all_feats
                ]
                ret['id_loss'] = (
                    objectives.compute_id(logits_list, batch['pids'])
                    * self.args.id_loss_weight
                )
            
            if self.training:
                self.update_rgb_center(rgb_cls.detach())

            return ret


        nir_feats = self.encode_nir_cls(nirs)
        cp_feats  = self.encode_cp_cls(cps)
        sk_feats  = self.encode_sk_cls(sks)
        text_feats = self.encode_text_cls(texts)

        if 'itc' in self.current_task:
            ret.update({'nir_itc_loss': objectives.compute_itc(rgb_cls, nir_feats, logit_scale)})
            ret.update({'cp_itc_loss': objectives.compute_itc(rgb_cls, cp_feats, logit_scale)})
            ret.update({'sk_itc_loss': objectives.compute_itc(rgb_cls, sk_feats, logit_scale)})
            ret.update({'txt_itc_loss': objectives.compute_itc(rgb_cls, text_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'nir_sdm_loss': objectives.compute_sdm(rgb_cls, nir_feats, batch['pids'], logit_scale)})
            ret.update({'cp_sdm_loss': objectives.compute_sdm(rgb_cls, cp_feats, batch['pids'], logit_scale)})
            ret.update({'sk_sdm_loss': objectives.compute_sdm(rgb_cls, sk_feats, batch['pids'], logit_scale)})
            ret.update({'txt_sdm_loss': objectives.compute_sdm(rgb_cls, text_feats, batch['pids'], logit_scale)})

        return ret


def build_model_v1(args, num_classes=11003):
    model = ReID5oModel(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model