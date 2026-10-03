"""Verboa Video: the whole LTX-2.5 image-to-video stage behind one node.

It calls ComfyUI's own LTX nodes in the order of the official two-stage LTX-2.5 image-to-video template, with the settings our
clips use: the distilled transformer, 8 steps at half resolution, the x2 latent upscaler, 3 steps at full resolution, sound
generated together with the picture. Nothing here is a new sampler: every step is a ComfyUI node called by name, so the result is
what the same graph built by hand would give. The loaded models are kept between runs, so only the first run pays for loading.
"""
from __future__ import annotations

import gc
import logging
import math

import folder_paths
import nodes
import comfy.utils
from comfy_api.latest import io

LOG =logging.getLogger("verboa-video")

NONE_FOUND = "(none found, see the README for the file and its folder)"
STAGE1_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"   # 8 steps at half resolution
STAGE2_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"                                               # 3 steps at full resolution
STAGE2_SEED = 42                                                                          # as in the template
ASPECTS = {
    "match image": None,
    "2:3 portrait": (2, 3), "3:4 portrait": (3, 4), "9:16 portrait": (9, 16),
    "1:1 square": (1, 1),
    "4:3 landscape": (4, 3), "3:2 landscape": (3, 2), "16:9 landscape": (16, 9),
}


def _names(kind: str) -> list[str]:
    found = folder_paths.get_filename_list(kind)
    return list(found) if found else [NONE_FOUND]


def _pick(options: list[str], *want: str) -> str:
    """The first option whose name has all of the words in `want` (lower case), else the first option."""
    for name in options:
        low = name.lower()
        if all(w in low for w in want):
            return name
    return options[0]


def call(name: str, **kwargs):
    """Run a ComfyUI node by its registered name and return its outputs as a tuple (works for legacy and V3 nodes)."""
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f"This ComfyUI has no '{name}' node. Update ComfyUI: the LTX-2.5 nodes need a recent version.")
    target = cls if hasattr(cls, "EXECUTE_NORMALIZED") else cls()
    result = getattr(target, cls.FUNCTION)(**kwargs)
    if hasattr(result, "args") and not isinstance(result, tuple):   # a V3 NodeOutput
        return tuple(result.args)
    if isinstance(result, dict) and "result" in result:
        return tuple(result["result"])
    return result if isinstance(result, tuple) else (result,)


_CACHE: dict = {"key": None, "models": None}


def load_models(unet: str, clip: str, vae: str, audio_vae: str, upscaler: str):
    """The five model files, loaded once and kept. Choosing different files replaces them."""
    for label, value in (("transformer", unet), ("text encoder", clip), ("video VAE", vae), ("audio VAE", audio_vae), ("upscaler", upscaler)):
        if value == NONE_FOUND:
            raise RuntimeError(f"No {label} file found. Put the LTX-2.5 {label} file in its ComfyUI models folder (see the README), "
                               "refresh the node definitions (press R) and choose it.")
    key = (unet, clip, vae, audio_vae, upscaler)
    if _CACHE["key"] == key and _CACHE["models"] is not None:
        return _CACHE["models"]
    _CACHE["key"], _CACHE["models"] = None, None
    gc.collect()
    model = call("UNETLoader", unet_name=unet, weight_dtype="default")[0]
    text_encoder = call("CLIPLoader", clip_name=clip, type="ltxv", device="default")[0]
    video_vae = call("VAELoader", vae_name=vae)[0]
    sound_vae = call("VAELoader", vae_name=audio_vae)[0]
    up = call("LatentUpscaleModelLoader", model_name=upscaler)[0]
    _CACHE["key"], _CACHE["models"] = key, (model, text_encoder, video_vae, sound_vae, up)
    return _CACHE["models"]


def resolution(img_w: int, img_h: int, aspect: str, megapixels: float) -> tuple[int, int]:
    """Output size: the aspect (or the image's own) at about `megapixels`, both sides a multiple of 64 so that the half
    resolution first stage is a multiple of 32, which the LTX latents need."""
    ratio = ASPECTS.get(aspect)
    ar = (img_w / img_h) if ratio is None else ratio[0] / ratio[1]
    w = math.sqrt(megapixels * 1_000_000 * ar)
    h = w / ar
    return max(256, int(round(w / 64)) * 64), max(256, int(round(h / 64)) * 64)


def frame_count(duration: float, fps: int) -> int:
    """LTX wants 8n+1 frames; the nearest to duration x fps."""
    return max(8, int(round(duration * fps / 8)) * 8) + 1


def prep_image(image, longer_side: int = 1536):
    """Scale the longer side to 1536 with Lanczos, as the template does before the LTX preprocess."""
    _, h, w, _ = image.shape
    scale = longer_side / max(h, w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    x = comfy.utils.common_upscale(image.movedim(-1, 1), nw, nh, "lanczos", "disabled")
    return x.movedim(1, -1)


def generate(image, prompt, negative, models, w, h, frames, fps, seed, video_cfg, audio_cfg, first_frame_strength, compression, sigmas1, sigmas2):
    model, clip, vae, audio_vae, upscaler = models
    positive = call("CLIPTextEncode", clip=clip, text=prompt)[0]
    negative_c = call("CLIPTextEncode", clip=clip, text=negative)[0]
    positive, negative_c = call("LTXVConditioning", positive=positive, negative=negative_c, frame_rate=float(fps))
    first = call("LTXVPreprocess", image=prep_image(image), img_compression=int(compression))[0]
    sampler = call("KSamplerSelect", sampler_name="euler_ancestral")[0]

    # stage 1: half resolution, 8 steps, picture and sound together
    latent = call("EmptyLTXVLatentVideo", width=w // 2, height=h // 2, length=frames, batch_size=1)[0]
    latent = call("LTXVImgToVideoInplace", vae=vae, image=first, latent=latent, strength=float(first_frame_strength), bypass=False)[0]
    sound = call("LTXVEmptyLatentAudio", frames_number=frames, frame_rate=float(fps), batch_size=1, audio_vae=audio_vae)[0]
    packed = call("LTXVConcatAVLatent", video_latent=latent, audio_latent=sound)[0]
    guider = call("LTXVDualCFGGuider", model=model, positive=positive, negative=negative_c, video_cfg=float(video_cfg), audio_cfg=float(audio_cfg))[0]
    noise = call("RandomNoise", noise_seed=int(seed))[0]
    sigmas = call("ManualSigmas", sigmas=sigmas1)[0]
    out = call("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=packed)[0]
    video_latent, sound = call("LTXVSeparateAVLatent", av_latent=out)

    # stage 2: x2 latent upscale, first frame pinned again, 3 steps at full resolution
    video_latent = call("LTXVLatentUpsampler", samples=video_latent, upscale_model=upscaler, vae=vae)[0]
    video_latent = call("LTXVImgToVideoInplace", vae=vae, image=first, latent=video_latent, strength=1.0, bypass=False)[0]
    packed = call("LTXVConcatAVLatent", video_latent=video_latent, audio_latent=sound)[0]
    guider = call("LTXVDualCFGGuider", model=model, positive=positive, negative=negative_c, video_cfg=float(video_cfg), audio_cfg=float(audio_cfg))[0]
    noise = call("RandomNoise", noise_seed=STAGE2_SEED)[0]
    sigmas = call("ManualSigmas", sigmas=sigmas2)[0]
    out = call("SamplerCustomAdvanced", noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=packed)[0]
    video_latent, sound = call("LTXVSeparateAVLatent", av_latent=out)

    pictures = call("VAEDecodeTiled", samples=video_latent, vae=vae, tile_size=512, overlap=64, temporal_size=64, temporal_overlap=16)[0]
    audio = call("LTXVAudioVAEDecode", samples=sound, audio_vae=audio_vae)[0]
    video = call("CreateVideo", images=pictures, fps=float(fps), audio=audio, bit_depth=8, color_space="sRGB", codec="none")[0]
    return video, pictures


class VerboaVideoI2V(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        unets, clips = _names("diffusion_models"), _names("text_encoders")
        vaes, ups = _names("vae"), _names("latent_upscale_models")
        return io.Schema(
            node_id="VerboaVideoI2V",
            display_name="Verboa Video (LTX-2.5 image to video)",
            category="Verboa",
            description="Turns one picture into a video with sound: LTX-2.5 distilled, two stages (8 steps at half resolution, "
                        "x2 latent upscale, 3 steps at full resolution), exactly as in the official LTX-2.5 image-to-video "
                        "template. Describe what moves in the prompt.",
            inputs=[
                io.Image.Input("image", tooltip="The first frame. Frames where the subject faces the camera and the whole scene is in view animate best."),
                io.String.Input("prompt", multiline=True, default="",
                                tooltip="What happens: who moves how, the camera, the sounds."),
                io.Int.Input("duration", default=6, min=2, max=10, tooltip="Length in seconds."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.randomize,
                             tooltip="Randomized after every run."),
                io.Combo.Input("aspect", options=list(ASPECTS), default="match image", advanced=True,
                               tooltip="Shape of the video. 'match image' keeps the first frame's own shape (no cropping)."),
                io.Float.Input("megapixels", default=0.9, min=0.25, max=2.0, step=0.05, advanced=True,
                               tooltip="Size of the video. 0.9 gives 768x1152 for a 2:3 frame (about 1 minute on an RTX 5090 laptop)."),
                io.Int.Input("fps", default=24, min=12, max=60, advanced=True, tooltip="Frames per second."),
                io.String.Input("negative_prompt", multiline=True, default="", advanced=True,
                                tooltip="Not used at the distilled settings (CFG 1); only matters if you raise the CFG."),
                io.Combo.Input("transformer", options=unets, default=_pick(unets, "ltx", "2.5", "distilled"), advanced=True,
                               tooltip="models/diffusion_models: ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
                io.Combo.Input("text_encoder", options=clips, default=_pick(clips, "gemma", "ltx"), advanced=True,
                               tooltip="models/text_encoders: gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
                io.Combo.Input("video_vae", options=vaes, default=_pick(vaes, "ltx", "video"), advanced=True,
                               tooltip="models/vae: ltx-2.5-video-vae-bf16.safetensors"),
                io.Combo.Input("audio_vae", options=vaes, default=_pick(vaes, "ltx", "audio"), advanced=True,
                               tooltip="models/vae: ltx-2.5-audio-vae-bf16.safetensors"),
                io.Combo.Input("upscaler", options=ups, default=_pick(ups, "ltx", "upscaler"), advanced=True,
                               tooltip="models/latent_upscale_models: ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
                io.Float.Input("first_frame_strength", default=0.7, min=0.0, max=1.0, step=0.05, advanced=True,
                               tooltip="How firmly the first stage holds the first frame (the template uses 0.7)."),
                io.Int.Input("img_compression", default=18, min=0, max=100, advanced=True,
                             tooltip="LTX preprocess: compression artifacts added to the first frame so it matches the training data (template: 18)."),
                io.Float.Input("video_cfg", default=1.0, min=0.0, max=20.0, step=0.1, advanced=True, tooltip="1.0 for the distilled model."),
                io.Float.Input("audio_cfg", default=1.0, min=0.0, max=20.0, step=0.1, advanced=True, tooltip="1.0 for the distilled model."),
                io.String.Input("stage1_sigmas", default=STAGE1_SIGMAS, advanced=True, tooltip="The 8 steps at half resolution."),
                io.String.Input("stage2_sigmas", default=STAGE2_SIGMAS, advanced=True, tooltip="The 3 steps at full resolution."),
            ],
            outputs=[
                io.Video.Output(display_name="video"),
                io.Image.Output(display_name="frames", tooltip="Every frame, for the Verboa Frame Sheet node or your own checks."),
                io.String.Output(display_name="settings"),
            ],
        )

    @classmethod
    def execute(cls, image, prompt, duration, seed, aspect, megapixels, fps, negative_prompt, transformer, text_encoder, video_vae,
                audio_vae, upscaler, first_frame_strength, img_compression, video_cfg, audio_cfg, stage1_sigmas, stage2_sigmas) -> io.NodeOutput:
        if not prompt.strip():
            raise ValueError("Write a prompt that says what moves (who moves how, the camera, the sounds), or connect a text output to the prompt input.")
        if image.shape[0] > 1:
            LOG.warning("Verboa Video: the image input holds %d images; only the first is used.", image.shape[0])
        image = image[:1]
        models = load_models(transformer, text_encoder, video_vae, audio_vae, upscaler)
        w, h = resolution(image.shape[2], image.shape[1], aspect, megapixels)
        frames = frame_count(duration, fps)
        video, pictures = generate(image, prompt, negative_prompt, models, w, h, frames, fps, seed, video_cfg, audio_cfg,
                                   first_frame_strength, img_compression, stage1_sigmas, stage2_sigmas)
        settings = (f"LTX-2.5 distilled image to video | {w}x{h} | {frames} frames at {fps} fps ({frames / fps:.1f} s) | seed {seed} | "
                    f"{len(stage1_sigmas.split(',')) - 1} + {len(stage2_sigmas.split(',')) - 1} steps")
        return io.NodeOutput(video, pictures, settings)
