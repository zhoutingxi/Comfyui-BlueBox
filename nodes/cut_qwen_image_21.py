import math
import torch

# Qwen-Image 2.1 编辑训练桶常见比例，用于语义一致性打分
_STANDARD_RATIOS = [
    1.0,          # 1:1
    4 / 3,        # 4:3
    3 / 4,        # 3:4
    16 / 9,       # 16:9
    9 / 16,       # 9:16
    3 / 2,        # 3:2
    2 / 3,        # 2:3
]

# resolution 合法档位：512 ~ 2048，步进 32
_RESOLUTIONS = list(range(512, 2048 + 1, 32))


def _auto_resolution(width: int, height: int) -> int:
    """
    根据输入图面积，自动对齐到最接近的合法 resolution。
    规则：取 sqrt(w*h) 的最近邻 32 倍数，夹在 [_RESOLUTIONS[0], _RESOLUTIONS[-1]]。
    """
    target = math.sqrt(float(width) * float(height))
    snapped = int(round(target / 32.0)) * 32
    lo, hi = _RESOLUTIONS[0], _RESOLUTIONS[-1]
    if snapped < lo:
        snapped = lo
    if snapped > hi:
        snapped = hi
    if snapped not in _RESOLUTIONS:
        snapped = min(_RESOLUTIONS, key=lambda r: abs(r - snapped))
    return snapped


def _generate_whitelist(resolution: int, tol_ratio: float = 1.0):
    """
    生成 resolution 对应的“不动点”尺寸白名单。
    规则：宽高均为 32 的倍数，k*m 落在 n^2 ± tol 内（n = resolution // 32）。
    返回 [(w, h, prod), ...]
    """
    n = resolution // 32
    if n < 1:
        return []

    target = n * n
    tol = max(1, int(n * tol_ratio))

    pairs = []
    seen = set()
    for k in range(1, n * 3):
        for m in range(1, n * 3):
            prod = k * m
            if abs(prod - target) <= tol:
                w, h = k * 32, m * 32
                if (w, h) in seen:
                    continue
                seen.add((w, h))
                pairs.append((w, h, prod))
    return pairs


def _ratio_distance(a: float, b: float) -> float:
    """对数比例距离，对称且符合比例感知。"""
    return abs(math.log(a) - math.log(b))


def _score_candidate(w, h, prod, resolution, input_ratio):
    """
    综合打分：
    - 比例与输入图的偏离（裁切损失）
    - 比例偏离标准训练桶的惩罚（语义一致性风险）
    - 面积偏离 resolution^2 的惩罚（对齐误差风险）
    分数越低越好。
    """
    ratio = w / h

    d_input = _ratio_distance(ratio, input_ratio)
    d_std = min(_ratio_distance(ratio, r) for r in _STANDARD_RATIOS)

    target_area = resolution * resolution
    d_area = abs(prod * 1024 - target_area) / target_area  # prod 是 32x32 格数

    return d_input * 1.0 + d_std * 0.6 + d_area * 0.4


def _pick_target_size(width, height, resolution, tol_ratio=1.0):
    input_ratio = width / height
    whitelist = _generate_whitelist(resolution, tol_ratio)
    if not whitelist:
        raise ValueError(f"resolution={resolution} 无法生成白名单")

    best = None
    best_score = float("inf")
    for w, h, prod in whitelist:
        s = _score_candidate(w, h, prod, resolution, input_ratio)
        if s < best_score:
            best_score = s
            best = (w, h)
    return best, whitelist


def _verify_fixed_point(w, h, resolution):
    """
    模拟 Qwen-Image 2.1 内部缩放：缩放到约 resolution^2 面积、保持比例、对齐 32。
    返回经过内部缩放后的 (w2, h2)。若等于 (w, h)，则为不动点。
    """
    scale_by = math.sqrt((resolution * resolution) / (w * h))
    w2 = max(32, int(round(w * scale_by / 32.0)) * 32)
    h2 = max(32, int(round(h * scale_by / 32.0)) * 32)
    return w2, h2


def _center_crop_to(image: torch.Tensor, tw: int, th: int):
    """
    image: [B, H, W, C] 或 [H, W, C]
    保持宽高比缩放覆盖目标，再中心裁切到 (tw, th)。
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True

    b, h, w, c = image.shape

    scale = max(tw / w, th / h)
    nw = max(tw, int(round(w * scale)))
    nh = max(th, int(round(h * scale)))

    img = image.permute(0, 3, 1, 2)  # BCHW
    img = torch.nn.functional.interpolate(
        img, size=(nh, nw), mode="bilinear", align_corners=False
    )
    img = img.permute(0, 2, 3, 1)  # BHWC

    top = (nh - th) // 2
    left = (nw - tw) // 2
    img = img[:, top:top + th, left:left + tw, :]

    if squeeze:
        img = img.squeeze(0)
    return img


def _pad_to(image: torch.Tensor, tw: int, th: int, pad_color: float = 0.0):
    """
    image: [B, H, W, C] 或 [H, W, C]
    保持宽高比缩放至完全放入 (tw, th)，再居中补边到目标尺寸。
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True

    b, h, w, c = image.shape

    scale = min(tw / w, th / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))

    img = image.permute(0, 3, 1, 2)
    img = torch.nn.functional.interpolate(
        img, size=(nh, nw), mode="bilinear", align_corners=False
    )
    img = img.permute(0, 2, 3, 1)

    canvas = torch.full((b, th, tw, c), pad_color, dtype=img.dtype, device=img.device)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[:, top:top + nh, left:left + nw, :] = img

    if squeeze:
        canvas = canvas.squeeze(0)
    return canvas


def _stretch_to(image: torch.Tensor, tw: int, th: int):
    """
    image: [B, H, W, C] 或 [H, W, C]
    直接双线性插值拉伸到 (tw, th)，不保持宽高比。
    """
    squeeze = False
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze = True

    img = image.permute(0, 3, 1, 2)  # BCHW
    img = torch.nn.functional.interpolate(
        img, size=(th, tw), mode="bilinear", align_corners=False
    )
    img = img.permute(0, 2, 3, 1)  # BHWC

    if squeeze:
        img = img.squeeze(0)
    return img


def _snap_fixed_point(tw, th, resolution, tol_ratio, input_w, input_h):
    """
    若 (tw, th) 不是严格不动点，回退到白名单中比例最接近输入、且校验通过的一对。
    """
    w2, h2 = _verify_fixed_point(tw, th, resolution)
    if (w2, h2) == (tw, th):
        return tw, th

    whitelist = _generate_whitelist(resolution, tol_ratio)
    input_ratio = input_w / input_h
    fallback = None
    best_err = float("inf")
    for cw, ch, _ in whitelist:
        cw2, ch2 = _verify_fixed_point(cw, ch, resolution)
        if (cw2, ch2) == (cw, ch):
            err = _ratio_distance(cw / ch, input_ratio)
            if err < best_err:
                best_err = err
                fallback = (cw, ch)

    if fallback:
        return fallback
    return tw, th


class CutQwenImage21:
    """
    Cut (Qwen-Image 2.1) — BlueBox
    将任意尺寸图像约束裁切到 Qwen-Image 2.1 在当前 resolution 下的“不动点”标准尺寸。

    resolution:
      - default : 根据输入图面积自动对齐到最接近的合法档位（sqrt(w*h) 最近邻 32 倍数）
      - 512~2048: 手动指定档位

    mode:
      - crop    : 保持比例缩放覆盖，中心裁切（比例最接近输入，会有裁切）
      - pad     : 保持比例缩放内嵌，居中补边（不裁切，可能有留白）
      - stretch : 直接拉伸到目标尺寸（不裁切不留白，比例偏离输入时会变形）
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "resolution": (
                    ["default"] + [str(r) for r in _RESOLUTIONS],
                    {"default": "default"},
                ),
                "tol_ratio": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.2,
                    "max": 3.0,
                    "step": 0.1,
                }),
                "mode": (["crop", "pad", "stretch"], {"default": "crop"}),
                "pad_color": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height", "resolution")
    FUNCTION = "run"
    CATEGORY = "BlueBox/Qwen"

    def run(self, image, resolution, tol_ratio, mode, pad_color):
        _, h, w, _ = image.shape

        # resolution 兼容三种输入：
        #   - "default"      → 根据输入图面积自动对齐
        #   - 字符串数字     → 转 int（新下拉）
        #   - 整数           → 直接用（老工作流）
        if resolution == "default" or resolution is None:
            resolution = _auto_resolution(w, h)
        else:
            resolution = int(resolution)

        (tw, th), _ = _pick_target_size(w, h, resolution, tol_ratio)
        tw, th = _snap_fixed_point(tw, th, resolution, tol_ratio, w, h)

        if mode == "pad":
            out = _pad_to(image, tw, th, pad_color)
        elif mode == "stretch":
            out = _stretch_to(image, tw, th)
        else:
            out = _center_crop_to(image, tw, th)

        return (out, tw, th, resolution)


NODE_CLASS_MAPPINGS = {
    "CutQwenImage21": CutQwenImage21,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CutQwenImage21": "Cut (Qwen-Image 2.1) — BlueBox",
}