# Notices

This folder holds ComfyUI custom nodes (code) and two example workflows. It contains no model weights.

| component | author | license | how it is used here |
|---|---|---|---|
| ComfyUI and its LTX, sampler, VAE and video nodes | Comfy Org and contributors | GPL-3.0 | The Verboa Video node calls ComfyUI's own nodes by name at run time. None of their code is copied into this folder. |
| LTX-2.5 (distilled transformer, Gemma text encoder, video and audio VAEs, latent upscaler) | Lightricks | LTX-2.x Community License | Downloaded by you from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5). Not distributed here. Their license applies to the files and to what you make with them. |
| Verboa Image 1.0 | Verboa AI | CreativeML Open RAIL++-M with one additional use restriction | The first-frame model in `verboa-video-from-text.json`. Distributed separately on Hugging Face under its own license. |

The code in this folder is licensed by Verboa AI (verboa.io) under the license in `LICENSE.md`.
