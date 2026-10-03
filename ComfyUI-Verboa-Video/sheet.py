"""Verboa Frame Sheet: a strip of evenly spaced frames from a video, for a quick look before you post."""
from __future__ import annotations

import torch
import comfy.utils
from comfy_api.latest import io


class VerboaFrameSheet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VerboaFrameSheet",
            display_name="Verboa Frame Sheet",
            category="Verboa",
            description="One strip of evenly spaced frames from a video, for a quick look before you post: is the face the same all the way, "
                        "is the scene still in view, did a second head appear. Connect the frames output of Verboa Video and a Preview Image node.",
            inputs=[
                io.Image.Input("frames", tooltip="The frames output of Verboa Video."),
                io.Int.Input("count", default=5, min=2, max=16, tooltip="How many frames."),
                io.Int.Input("height", default=360, min=96, max=1024, tooltip="Height of each frame in the strip."),
            ],
            outputs=[io.Image.Output(display_name="sheet")],
        )

    @classmethod
    def execute(cls, frames, count, height) -> io.NodeOutput:
        n = frames.shape[0]
        picks = [min(n - 1, int(n * (i + 0.5) / count)) for i in range(count)]
        tiles = []
        for i in picks:
            f = frames[i:i + 1]
            w = max(1, round(f.shape[2] * height / f.shape[1]))
            tiles.append(comfy.utils.common_upscale(f.movedim(-1, 1), w, height, "area", "disabled").movedim(1, -1))
        return io.NodeOutput(torch.cat(tiles, dim=2))
