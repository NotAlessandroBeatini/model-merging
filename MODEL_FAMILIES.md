# OpenCLIP ViT Family + Shape Inventory

These inventories are generated from `open_clip.create_model(..., pretrained="")` with the text transformer removed (equivalent to `keep_lang: false` in this repo).

Note: TSV/Isotropic process **2D** params per-layer; `text_projection` is explicitly skipped in code.

## ViT-B-32

**2D parameter families (TSV/Isotropic SVD path; except `text_projection` is skipped in code)**
- `positional_embedding` — shapes: `(77, 512)` (params: 1)
- `text_projection` — shapes: `(512, 512)` (params: 1)
- `token_embedding.weight` — shapes: `(49408, 512)` (params: 1)
- `visual.positional_embedding` — shapes: `(50, 768)` (params: 1)
- `visual.proj` — shapes: `(768, 512)` (params: 1)
- `visual.transformer.resblocks.*.attn.in_proj_weight` — shapes: `(2304, 768)` (params: 12)
- `visual.transformer.resblocks.*.attn.out_proj.weight` — shapes: `(768, 768)` (params: 12)
- `visual.transformer.resblocks.*.mlp.c_fc.weight` — shapes: `(3072, 768)` (params: 12)
- `visual.transformer.resblocks.*.mlp.c_proj.weight` — shapes: `(768, 3072)` (params: 12)

**Non-2D parameter families (averaged/kept base depending on config)**
- `ln_final.bias` — shapes: `(512,)` (params: 1)
- `ln_final.weight` — shapes: `(512,)` (params: 1)
- `logit_scale` — shapes: `()` (params: 1)
- `visual.class_embedding` — shapes: `(768,)` (params: 1)
- `visual.conv1.weight` — shapes: `(768, 3, 32, 32)` (params: 1)
- `visual.ln_post.bias` — shapes: `(768,)` (params: 1)
- `visual.ln_post.weight` — shapes: `(768,)` (params: 1)
- `visual.ln_pre.bias` — shapes: `(768,)` (params: 1)
- `visual.ln_pre.weight` — shapes: `(768,)` (params: 1)
- `visual.transformer.resblocks.*.attn.in_proj_bias` — shapes: `(2304,)` (params: 12)
- `visual.transformer.resblocks.*.attn.out_proj.bias` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.ln_1.bias` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.ln_1.weight` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.ln_2.bias` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.ln_2.weight` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.mlp.c_fc.bias` — shapes: `(3072,)` (params: 12)
- `visual.transformer.resblocks.*.mlp.c_proj.bias` — shapes: `(768,)` (params: 12)

## ViT-B-16

**2D parameter families (TSV/Isotropic SVD path; except `text_projection` is skipped in code)**
- `positional_embedding` — shapes: `(77, 512)` (params: 1)
- `text_projection` — shapes: `(512, 512)` (params: 1)
- `token_embedding.weight` — shapes: `(49408, 512)` (params: 1)
- `visual.positional_embedding` — shapes: `(197, 768)` (params: 1)
- `visual.proj` — shapes: `(768, 512)` (params: 1)
- `visual.transformer.resblocks.*.attn.in_proj_weight` — shapes: `(2304, 768)` (params: 12)
- `visual.transformer.resblocks.*.attn.out_proj.weight` — shapes: `(768, 768)` (params: 12)
- `visual.transformer.resblocks.*.mlp.c_fc.weight` — shapes: `(3072, 768)` (params: 12)
- `visual.transformer.resblocks.*.mlp.c_proj.weight` — shapes: `(768, 3072)` (params: 12)

**Non-2D parameter families (averaged/kept base depending on config)**
- `ln_final.bias` — shapes: `(512,)` (params: 1)
- `ln_final.weight` — shapes: `(512,)` (params: 1)
- `logit_scale` — shapes: `()` (params: 1)
- `visual.class_embedding` — shapes: `(768,)` (params: 1)
- `visual.conv1.weight` — shapes: `(768, 3, 16, 16)` (params: 1)
- `visual.ln_post.bias` — shapes: `(768,)` (params: 1)
- `visual.ln_post.weight` — shapes: `(768,)` (params: 1)
- `visual.ln_pre.bias` — shapes: `(768,)` (params: 1)
- `visual.ln_pre.weight` — shapes: `(768,)` (params: 1)
- `visual.transformer.resblocks.*.attn.in_proj_bias` — shapes: `(2304,)` (params: 12)
- `visual.transformer.resblocks.*.attn.out_proj.bias` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.ln_1.bias` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.ln_1.weight` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.ln_2.bias` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.ln_2.weight` — shapes: `(768,)` (params: 12)
- `visual.transformer.resblocks.*.mlp.c_fc.bias` — shapes: `(3072,)` (params: 12)
- `visual.transformer.resblocks.*.mlp.c_proj.bias` — shapes: `(768,)` (params: 12)

## ViT-L-14

**2D parameter families (TSV/Isotropic SVD path; except `text_projection` is skipped in code)**
- `positional_embedding` — shapes: `(77, 768)` (params: 1)
- `text_projection` — shapes: `(768, 768)` (params: 1)
- `token_embedding.weight` — shapes: `(49408, 768)` (params: 1)
- `visual.positional_embedding` — shapes: `(257, 1024)` (params: 1)
- `visual.proj` — shapes: `(1024, 768)` (params: 1)
- `visual.transformer.resblocks.*.attn.in_proj_weight` — shapes: `(3072, 1024)` (params: 24)
- `visual.transformer.resblocks.*.attn.out_proj.weight` — shapes: `(1024, 1024)` (params: 24)
- `visual.transformer.resblocks.*.mlp.c_fc.weight` — shapes: `(4096, 1024)` (params: 24)
- `visual.transformer.resblocks.*.mlp.c_proj.weight` — shapes: `(1024, 4096)` (params: 24)

**Non-2D parameter families (averaged/kept base depending on config)**
- `ln_final.bias` — shapes: `(768,)` (params: 1)
- `ln_final.weight` — shapes: `(768,)` (params: 1)
- `logit_scale` — shapes: `()` (params: 1)
- `visual.class_embedding` — shapes: `(1024,)` (params: 1)
- `visual.conv1.weight` — shapes: `(1024, 3, 14, 14)` (params: 1)
- `visual.ln_post.bias` — shapes: `(1024,)` (params: 1)
- `visual.ln_post.weight` — shapes: `(1024,)` (params: 1)
- `visual.ln_pre.bias` — shapes: `(1024,)` (params: 1)
- `visual.ln_pre.weight` — shapes: `(1024,)` (params: 1)
- `visual.transformer.resblocks.*.attn.in_proj_bias` — shapes: `(3072,)` (params: 24)
- `visual.transformer.resblocks.*.attn.out_proj.bias` — shapes: `(1024,)` (params: 24)
- `visual.transformer.resblocks.*.ln_1.bias` — shapes: `(1024,)` (params: 24)
- `visual.transformer.resblocks.*.ln_1.weight` — shapes: `(1024,)` (params: 24)
- `visual.transformer.resblocks.*.ln_2.bias` — shapes: `(1024,)` (params: 24)
- `visual.transformer.resblocks.*.ln_2.weight` — shapes: `(1024,)` (params: 24)
- `visual.transformer.resblocks.*.mlp.c_fc.bias` — shapes: `(4096,)` (params: 24)
- `visual.transformer.resblocks.*.mlp.c_proj.bias` — shapes: `(1024,)` (params: 24)
