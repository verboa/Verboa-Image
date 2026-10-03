# Verboa Video for ComfyUI

<p align="center">
  <a href="https://verboa.io">verboa.io</a> &nbsp;|&nbsp;
  <a href="https://huggingface.co/verboa/Verboa-Image-1.0">Hugging Face</a> &nbsp;|&nbsp;
  <a href="https://civitai.red/user/verboa">CivitAI</a> &nbsp;|&nbsp;
  <a href="https://x.com/verboa_ai">X</a>
</p>

The video stage of the pipeline we make our own clips with, as ComfyUI nodes: one picture in, a video **with sound** out, made by
[LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5). There is no graph to build: connect a picture, write what moves, press Run. We use it after
[Verboa Image 1.0](https://huggingface.co/verboa/Verboa-Image-1.0), which draws the first frame, but it works with any picture.

> **18+ only.** Verboa Image generates explicit adult content on request. This page contains none, and neither does this folder.

```
Load Image ──► Verboa Video ──► Save Video
                    └──► Verboa Frame Sheet ──► Preview Image   (a strip of frames to look at before you post)
```

## The nodes

| node | what it does |
|---|---|
| **Verboa Video (LTX-2.5 image to video)** | The whole two-stage LTX-2.5 image-to-video pipeline in one node: first frame in, mp4 with sound out. It calls ComfyUI's own LTX nodes in the order of the official template (distilled transformer, 8 steps at half resolution, x2 latent upscaler, 3 steps at full resolution, picture and sound sampled together). Outputs `video`, every `frames` image and a `settings` line. |
| **Verboa Frame Sheet** | A strip of evenly spaced frames from the video (default 5), for a quick look before you post: is it the same face all the way, is the scene still in view, did a second head appear. |

Both are under the category **Verboa**. The loaded models stay in memory between runs, so only the first run pays for loading.

## Install

1. ComfyUI with the LTX-2.5 nodes (tested on 0.37.4; the node calls `LTXVDualCFGGuider`, `LTXVLatentUpsampler`, `LTXVImgToVideoInplace`,
   `LTXVEmptyLatentAudio`, `LTXVAudioVAEDecode` and `CreateVideo` by name, so any ComfyUI that has them works).
2. Copy the `ComfyUI-Verboa-Video` folder into `ComfyUI/custom_nodes/` (or unzip the release there) and restart ComfyUI. No extra Python packages.
3. Put the five LTX-2.5 files from [Lightricks/LTX-2.5](https://huggingface.co/Lightricks/LTX-2.5) (the repo is gated: accept the LTX-2.x
   Community License there). The folder names in that repo are the folder names under `ComfyUI/models/`:

| file | size | folder |
|---|---|---|
| `ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors` | 21.5 GB | `models/diffusion_models/` |
| `gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors` | 15.4 GB | `models/text_encoders/` |
| `ltx-2.5-video-vae-bf16.safetensors` | 1.5 GB | `models/vae/` |
| `ltx-2.5-audio-vae-bf16.safetensors` | 0.4 GB | `models/vae/` |
| `ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors` | 1.0 GB | `models/latent_upscale_models/` |

4. Load [`workflows/verboa-video-i2v.json`](workflows/verboa-video-i2v.json): Load Image, Verboa Video, Save Video, plus the frame sheet.

A third node, **Verboa Act Prompt**, writes the first-frame prompt and the motion prompt of a clip from a seed. It is distributed on
[Hugging Face](https://huggingface.co/verboa/Verboa-Prompt-Writer) with the Prompt Writer, because its prompts are explicit and GitHub does not allow that.

## Settings

Open **Advanced** on the node for the rest.

| setting | default | what it does |
|---|---|---|
| `duration` | 6 | seconds (2 to 10). The frame count is rounded to the 8n+1 LTX wants. |
| `seed` | random | randomized after every run |
| `aspect` | match image | the shape of the video; `match image` keeps the first frame's own shape. Fixed shapes: 2:3, 3:4, 9:16, 1:1, 4:3, 3:2, 16:9 |
| `megapixels` | 0.9 | size: 0.9 gives 768x1152 for a 2:3 frame. Both sides are rounded to a multiple of 64 (the half-resolution first stage needs multiples of 32) |
| `fps` | 24 | frames per second |
| `first_frame_strength` | 0.7 | how firmly the first stage holds the first frame (the template's value) |
| `img_compression` | 18 | the LTX preprocess's compression artifacts on the first frame (the template's value) |
| `video_cfg`, `audio_cfg` | 1.0 | 1.0 for the distilled model |
| `stage1_sigmas`, `stage2_sigmas` | 8 steps, 3 steps | the template's sigma lists |
| model files | the five above | choose other copies of the files if you have them |

Speed on an RTX 5090 laptop (24 GB): a 6 s 768x1152 clip took 70 s on the first run (models loading) and about a minute after that; a 4 s 1024x576 clip took 40 s.
If you run out of memory, lower `megapixels` (0.6 is fine) or `duration`.

## What we learned making clips

- **The first frame decides the clip.** Frames where the subject faces the camera and the whole scene is in view animate best. Views from behind drift, and tightly
  entangled poses tend to hide what is happening between the bodies.
- **One head per person.** Duplicated heads are the most common failure; say how many people are in the frame when you write the first-frame prompt.
- **Look at the strip before you post.** About half of our clips are keepers. We drop clips where the face changes, a second head shows up, or the camera drifts away from the subject.
- **The sound comes with the picture.** Say what you hear in the prompt (breathing, voices, the room) and LTX-2.5 generates it in the same pass.
- A prompt that says who moves how, what the camera does, and what you hear, in a few plain sentences, works better than a long list of tags.

## Posting clips

LTX-2.5's license (the LTX-2.x Community License) is free below USD 10 million revenue and asks that content which could be mistaken for real is marked as machine generated.
We put "AI-generated" in every title. Read the license on the model page before you post.

## License

The code in this folder is under the same license as Verboa Image: the CreativeML Open RAIL++-M License with one additional use restriction (`LICENSE.md`).
The LTX-2.5 files are not part of this folder; they come under Lightricks' license. See `NOTICE.md`.
