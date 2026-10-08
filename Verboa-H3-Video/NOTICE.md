# Notices

This folder holds two ComfyUI workflows (JSON) and their documentation. It contains no model weights and no code.

| component | author | license | how it is used here |
|---|---|---|---|
| ComfyUI and its core nodes (MiniMax H3, sampling, VAE, video) | Comfy Org and contributors | GPL-3.0 | The workflows call ComfyUI's own nodes by name. Nothing is copied into this folder. |
| MiniMax H3 (FL2VA transformer, Qwen3-VL text encoder, video and audio VAEs, 4-step turbo LoRA), repackaged for ComfyUI in [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) | MiniMax | [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE) | Downloaded by you. Not distributed here. Its license, territory terms and acceptable use policy apply to the files and to what you make with them. |
| ComfyUI-GGUF (Unet Loader (GGUF)) | city96 | Apache-2.0 | Loads the Verboa GGUF file in `verboa-h3-from-text.json`. Installed by you. |
| Verboa Image 1.0 | Verboa AI | CreativeML Open RAIL++-M with one additional use restriction | The first-frame model in `verboa-h3-from-text.json`. Distributed separately on Hugging Face under its own license. |
| Ministral 3 3B (text encoder), FLUX.2 VAE | Mistral AI, Black Forest Labs | Apache-2.0 | Used by Verboa Image in `verboa-h3-from-text.json`, from [Comfy-Org/ERNIE-Image](https://huggingface.co/Comfy-Org/ERNIE-Image). Not distributed here. |

MiniMax H3 is licensed under the MiniMax H3 Community License Agreement, Copyright © 2026 MiniMax. All Rights Reserved.

The workflows and documentation in this folder are licensed by Verboa AI (verboa.io) under the license in `LICENSE.md`.
