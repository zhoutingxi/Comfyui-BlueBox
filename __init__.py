import os
import folder_paths
from aiohttp import web
from server import PromptServer

from .bluebox_nodes import BlueBoxLoadImage, BlueBoxBridge
from . import bluebox_core as core

from .allpreview_nodes import (
    NODE_CLASS_MAPPINGS as PREVIEW_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
)

# === Comparison 节点 ===
from .comparison_nodes import (
    NODE_CLASS_MAPPINGS as COMPARE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COMPARE_NODE_DISPLAY_NAME_MAPPINGS,
)

from .nodes.rotate_image_bluebox import (
    NODE_CLASS_MAPPINGS as ROTATE_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ROTATE_NODE_DISPLAY_NAME_MAPPINGS,
)

# === Qwen-Image 2.1 约束裁切节点 ===
from .nodes.cut_qwen_image_21 import (
    NODE_CLASS_MAPPINGS as CUT_QWEN_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as CUT_QWEN_NODE_DISPLAY_NAME_MAPPINGS,
)

# === Color Calibrator 节点 ===
from .nodes.color_calibrator_bluebox import (
    NODE_CLASS_MAPPINGS as COLOR_CALIB_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COLOR_CALIB_NODE_DISPLAY_NAME_MAPPINGS,
)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "BlueBoxLoadImage": BlueBoxLoadImage,
    "BlueBoxBridge": BlueBoxBridge,
    **PREVIEW_NODE_CLASS_MAPPINGS,
    **COMPARE_NODE_CLASS_MAPPINGS,
    **ROTATE_NODE_CLASS_MAPPINGS,
    **CUT_QWEN_NODE_CLASS_MAPPINGS,
    **COLOR_CALIB_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlueBoxLoadImage": "Load Image (BlueBox)",
    "BlueBoxBridge": "Bridge (BlueBox)",
    **PREVIEW_NODE_DISPLAY_NAME_MAPPINGS,
    **COMPARE_NODE_DISPLAY_NAME_MAPPINGS,
    **ROTATE_NODE_DISPLAY_NAME_MAPPINGS,
    **CUT_QWEN_NODE_DISPLAY_NAME_MAPPINGS,
    **COLOR_CALIB_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["WEB_DIRECTORY", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]


@PromptServer.instance.routes.post("/bluebox/bridge_register")
async def bluebox_bridge_register(request):
    data = await request.json()
    node_id = str(data.get("node_id", ""))
    image_filename = data.get("image_filename")
    mask_filename = data.get("mask_filename")
    preview_filename = data.get("preview_filename")

    if not node_id or not image_filename:
        return web.json_response({"ok": False, "error": "missing node_id or image_filename"})

    input_dir = folder_paths.get_input_directory()

    def _resolve(fname):
        if not fname:
            return None
        p = os.path.join(input_dir, fname)
        return p if os.path.isfile(p) else None

    image_path = _resolve(image_filename)
    if image_path is None:
        return web.json_response({"ok": False, "error": "image file not found"})

    mask_path = _resolve(mask_filename)

    preview_item = None
    if preview_filename:
        p = os.path.join(input_dir, preview_filename)
        if os.path.isfile(p):
            preview_item = {
                "filename": preview_filename,
                "subfolder": "",
                "type": "input",
            }

    core.set_bridge_image(node_id, image_path, mask_path, preview_item)

    return web.json_response({"ok": True})


@PromptServer.instance.routes.post("/bluebox/bridge_continue")
async def bluebox_bridge_continue(request):
    data = await request.json()
    node_id = str(data.get("node_id", ""))

    if not node_id:
        return web.json_response({"ok": False, "error": "missing node_id"})

    core.set_bridge_continued(node_id, True)
    return web.json_response({"ok": True})


@PromptServer.instance.routes.get("/bluebox/bridge_current_image")
async def bluebox_bridge_current_image(request):
    node_id = str(request.query.get("node_id", ""))
    if not node_id:
        return web.json_response({"ok": False, "error": "missing node_id"})

    image_path, mask_path, preview_item = core.get_bridge_paths(node_id)
    if not image_path or not os.path.isfile(image_path):
        return web.json_response({"ok": False, "error": "no image"})

    if preview_item and preview_item.get("filename"):
        item = preview_item
    else:
        item = {
            "filename": os.path.basename(image_path),
            "subfolder": "",
            "type": "input",
        }

    return web.json_response({"ok": True, "image": item})


@PromptServer.instance.routes.get("/bluebox/bridge_reference_image")
async def bluebox_bridge_reference_image(request):
    node_id = str(request.query.get("node_id", ""))
    if not node_id:
        return web.json_response({"ok": False, "error": "missing node_id"})

    ref_path = core.get_bridge_reference(node_id)
    if not ref_path or not os.path.isfile(ref_path):
        return web.json_response({"ok": False, "error": "no reference"})

    return web.json_response({
        "ok": True,
        "image": {
            "filename": os.path.basename(ref_path),
            "subfolder": "BlueBoxBridge",
            "type": "temp",
        },
    })