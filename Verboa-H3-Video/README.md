# Verboa H3 Video: MiniMax H3 image-to-video workflows for ComfyUI

<p align="center">
  <a href="https://verboa.io">verboa.io</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/verboa/Verboa-Image-1.0">Hugging Face</a> &nbsp;|&nbsp;
  <a href="https://github.com/verboa/Verboa-Image">GitHub</a>
</p>

Two ComfyUI workflows that turn one still into a short clip **with sound**, using
[MiniMax H3](https://huggingface.co/Comfy-Org/MiniMax-H3). They are the video stage of the pipeline we make our own clips
with: the first frame comes from [Verboa Image 1.0](https://huggingface.co/verboa/Verboa-Image-1.0), or from any picture.
The video part uses core ComfyUI nodes only.

> **18+ only.** Verboa Image generates explicit adult content on request. This folder contains none.

| workflow | what it does |
|---|---|
| [`workflows/verboa-h3-i2v.json`](workflows/verboa-h3-i2v.json) | Load a picture, MiniMax H3 animates it, the mp4 with sound is saved. |
| [`workflows/verboa-h3-from-text.json`](workflows/verboa-h3-from-text.json) | The whole pipeline: Verboa Image 1.0 draws the first frame (8 steps), MiniMax H3 animates it, the still and the mp4 are saved. |

## What you need

1. ComfyUI with the core MiniMax H3 nodes (tested on 0.37.4).
2. The MiniMax H3 files from [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) (44.4 GB in total). They come
   under MiniMax's own license (see [License](#license)):

| file | size | folder |
|---|---|---|
| [`minimax_h3_fl2va_pruned_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors) | 20.97 GB | `models/diffusion_models/` |
| [`qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors) | 15.69 GB | `models/text_encoders/` |
| [`minimax_h3_video_vae_fp16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_video_vae_fp16.safetensors) | 5.21 GB | `models/vae/` |
| [`minimax_h3_audio_vae_fp32.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/vae/minimax_h3_audio_vae_fp32.safetensors) | 0.61 GB | `models/vae/` |
| [`minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors`](https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors) | 1.96 GB | `models/loras/` |

   The repo also has other precisions (int8 and bf16 text encoders, an int8 video VAE, fp8 and w6a8 transformers). The NVFP4
   text encoder is made for RTX 50-series cards.

3. For `verboa-h3-from-text.json` only, the first-frame model:

| file | size | folder |
|---|---|---|
| [`verboa-image-1.0-Q8_0.gguf`](https://huggingface.co/verboa/Verboa-Image-1.0-GGUF) (gated: accept the license on the page) | 8.71 GB | `models/diffusion_models/` |
| [`ministral-3-3b.safetensors`](https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/text_encoders/ministral-3-3b.safetensors) | 7.72 GB | `models/text_encoders/` |
| [`flux2-vae.safetensors`](https://huggingface.co/Comfy-Org/ERNIE-Image/resolve/main/vae/flux2-vae.safetensors) | 0.34 GB | `models/vae/` |

   The GGUF file loads with the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) custom node. Any Verboa GGUF works
   (Q4_K_M for 8 GB cards). With the fp8 or bf16 safetensors from [verboa/Verboa-Image-1.0](https://huggingface.co/verboa/Verboa-Image-1.0),
   swap the loader for the core *Load Diffusion Model* node.

The loader nodes carry the download links of the public files, the same way ComfyUI's own templates do.

## Use

1. Drag a workflow into ComfyUI.
2. `verboa-h3-i2v.json`: load your picture in **First frame**. `verboa-h3-from-text.json`: write the picture in **First-frame prompt**.
3. Describe the first frame and what happens next in **H3 prompt** (format below), set **Duration**, press Run.
   The clip lands in `output/video/`.

## Settings

| setting | default | what it does |
|---|---|---|
| Clip width, height | 512 x 896 (9:16) | Multiples of 32. 768 x 1344 is H3's native 768p size: sharper, but more than twice the render time (2.25x the pixels). The first frame is scaled and center-cropped to this size. |
| Duration | 6 s | Rounded up to H3's frame grid (17k+5 frames at 24 fps): 6 s gives 158 frames. H3 goes up to about 15 s. |
| Sampling | 4 steps | The 4-step turbo LoRA at 1.0, `res_multistep` sampler, `simple` scheduler, Basic Guider (no CFG). |
| Seed (H3) | random | A new take of the same first frame. |
| First frame (from-text) | 768 x 1344, 8 steps, CFG 2, shift 4 | Verboa Image's own settings, with an empty negative prompt (the model was trained with the empty caption as its negative). |

Speed on a laptop RTX 5090 (24 GB) with 64 GB of RAM: an 8 s clip at 512 x 896 takes about 110 s once the models are loaded,
a 12 s clip about 190 s. The first run also loads the 44 GB of H3 files.

## The H3 prompt

H3 reads a structured prompt. Keep the labels and the blank lines between them:

```
For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.

integrated_multimodal_description: [Shot 1] Live-action, vertical handheld phone footage, soft natural window light,
photorealistic skin, natural motion. At 00:00.000 a medium shot holds the exact starting frame from <Picture 1>: a woman in
her late twenties in a cream knit sweater, sitting at a small table by a sunny cafe window and looking out of the window.
From 00:01.200 she turns her head toward the camera, her hair swinging over her shoulder. From 00:02.400 she meets the lens
and breaks into a wide, warm smile, then laughs softly and tucks a strand of hair behind her ear. From 00:04.200 the camera
drifts slowly closer while she keeps smiling into the lens, relaxed and happy. The shot ends at 00:06.000. Nobody speaks:
no dialogue, only her soft laugh.

overall_soundscape: quiet cafe ambience, a low murmur of distant conversation, a cup set down on a saucer, her soft laugh,
faint handheld camera noise. No dialogue.

non_diegetic_music: N/A
```

- **Header**: `<Picture 1>` is your first frame.
- **integrated_multimodal_description**: `[Shot 1]`, the look of the footage, then the first frame exactly as it is
  (`At 00:00.000 ... from <Picture 1>: ...`), one time-coded beat per action (`From 00:02.000 ...`), and the end
  (`The shot ends at 00:06.000.`). Keep every time code inside the duration. A cut is `[Shot 2] At 00:04.000 hard cut to ...`;
  a spoken line is `The woman with a soft voice (S1) says: <d>[English] Hi.</d>`.
- **overall_soundscape**: everything you hear. H3 generates the sound in the same pass as the picture.
- **non_diegetic_music**: music that is not part of the scene, or `N/A`.

What works for us: a first frame with the subject facing the camera and fully in view, plain sentences with time codes
instead of tag lists, and handheld phone footage for a real look.

## More LoRAs

Extra H3 LoRAs (motion, style or NSFW LoRAs made for H3 FL2V) go after **Turbo LoRA**: add one *LoraLoaderModelOnly* node per
LoRA and connect the last one's MODEL to **Basic Guider** and **Scheduler**. Keep the turbo LoRA at 1.0 with 4 steps unless a
LoRA's author says otherwise, and put trigger words at the very start of the prompt.

## License

- **MiniMax H3 is not included and has its own license.** The workflows need the MiniMax H3 weights, which come under the
  [MiniMax H3 Community License Agreement](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE). Read it before you
  download them: among other terms it only covers use outside the United States, the European Union, the United Kingdom and
  South Korea, and it includes an acceptable use policy.
- Verboa Image 1.0 comes under the CreativeML Open RAIL++-M License with one additional use restriction (see its model card).
- The workflows and this documentation are released under the same license as Verboa Image (`LICENSE.md`). See `NOTICE.md`.
