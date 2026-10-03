"""Verboa Video for ComfyUI: the video stage of our pipeline as nodes (category "Verboa").

  Verboa Video          LTX-2.5 image to video with sound: the whole two-stage pipeline behind one node
  Verboa Frame Sheet    a strip of evenly spaced frames, for the look before you post
  Verboa Act Prompt     writes the first-frame prompt (for Verboa Image) and the motion prompt of one clip
                        (only present where actnode.py and acts.py are in the folder)

The pipeline: Verboa Image makes the first frame, Verboa Video animates it, SaveVideo writes the file. See the README for the
model files, the workflows and the settings. Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""
from __future__ import annotations

from comfy_api.latest import ComfyExtension, io

from .sheet import VerboaFrameSheet
from .video import VerboaVideoI2V

try:  # the act prompt node ships with some copies of this pack only
    from .actnode import VerboaActPrompt
except ImportError:
    VerboaActPrompt = None


class VerboaVideoExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        nodes = [VerboaVideoI2V, VerboaFrameSheet]
        if VerboaActPrompt is not None:
            nodes.insert(0, VerboaActPrompt)
        return nodes


async def comfy_entrypoint() -> VerboaVideoExtension:
    return VerboaVideoExtension()
