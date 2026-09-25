"""Generate one image with Verboa Image 1.0 in diffusers (NVIDIA GPU).

The pipeline calls are the model card's quick start: no prompt enhancer, an empty negative prompt, 8 steps at
guidance 2.0. The weights are gated: accept the terms on https://huggingface.co/verboa/Verboa-Image-1.0 while logged
in, then run `hf auth login` on this machine. On a Mac, use ComfyUI or mflux instead (see the README).

    pip install -r requirements.txt
    python inference.py --prompt "a candid phone photo of a woman in her twenties laughing at a cafe table, natural light"
"""
import argparse

import torch
from diffusers import ErnieImagePipeline

TRAINED_SIZES = ["1024x1024", "848x1264", "1264x848", "768x1376", "1376x768", "896x1200", "1200x896"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate one image with Verboa Image 1.0.")
    ap.add_argument("--prompt", default="a candid phone photo of a woman in her twenties laughing at a cafe table, "
                                        "natural light", help="write it the way you would say it")
    ap.add_argument("--size", default="848x1264",
                    help="WIDTHxHEIGHT, about 1 MP, multiples of 16; trained: " + ", ".join(TRAINED_SIZES))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="verboa.png")
    ap.add_argument("--model", default="verboa/Verboa-Image-1.0", help="Hugging Face repo id or a local folder")
    ap.add_argument("--offload", action="store_true", help="CPU offload: fits the bf16 model on a 24 GB card")
    args = ap.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    if width % 16 or height % 16:
        ap.error("width and height must be multiples of 16")

    pipe = ErnieImagePipeline.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        pe=None, pe_tokenizer=None,          # no prompt enhancer
    )
    if args.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    image = pipe(
        prompt=args.prompt,
        negative_prompt="",
        height=height, width=width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        use_pe=False,                        # required
        generator=torch.Generator("cuda").manual_seed(args.seed),
    ).images[0]
    image.save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
