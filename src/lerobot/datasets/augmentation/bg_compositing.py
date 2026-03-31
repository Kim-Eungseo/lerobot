"""Mask-based background texture compositing for online data augmentation."""

import os
import random

import cv2
import numpy as np


def _load_textures_by_region(texture_dir: str, resolution: int = 256) -> dict[str, list[np.ndarray]]:
    """Load textures from wall/, table/, floor/ subdirectories.

    Returns:
        {"wall": [ndarray(H,W,3), ...], "table": [...], "floor": [...]}
        Each array is RGB uint8, resized to resolution x resolution.
    """
    result: dict[str, list[np.ndarray]] = {"wall": [], "table": [], "floor": []}
    for region in result:
        region_dir = os.path.join(texture_dir, region)
        if not os.path.isdir(region_dir):
            continue
        for fname in sorted(os.listdir(region_dir)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(region_dir, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resolution, resolution))
            result[region].append(img)
    return result


class BgCompositor:
    """Mask-based background texture compositor.

    Textures are cached in memory at __init__ time.
    apply() is called per-sample from DatasetReader.get_item().
    """

    def __init__(
        self,
        texture_dir: str,
        bg_mode: str = "both",  # "paste" | "hsv" | "both"
        p_bg: float = 0.8,
        resolution: int = 256,
    ):
        self._textures = _load_textures_by_region(texture_dir, resolution)
        self._bg_mode = bg_mode
        self._p_bg = p_bg

    def has_textures(self) -> bool:
        return any(len(v) > 0 for v in self._textures.values())

    def apply(
        self,
        frames: np.ndarray,  # (T, H, W, 3) uint8 RGB
        masks: dict[str, np.ndarray],  # {"table": (T,H,W), "floor":..., "wall":...}
        table_tex_idx: int | None = None,
        floor_tex_idx: int | None = None,
        wall_tex_idx: int | None = None,
        mode: str | None = None,
    ) -> np.ndarray:
        """Apply background compositing.

        tex_idx parameters can be injected externally for domain-pair parameter fixing.
        None means random selection.
        """
        if mode is None:
            mode = random.choice(["paste", "hsv"]) if self._bg_mode == "both" else self._bg_mode

        result = frames.copy()
        h, w = result.shape[1], result.shape[2]

        def _pick(region: str, fixed_idx: int | None) -> tuple[np.ndarray, np.ndarray] | None:
            pool = self._textures[region]
            mask = masks.get(region)
            if not pool or mask is None:
                return None
            if random.random() > self._p_bg:
                return None
            idx = fixed_idx if fixed_idx is not None else random.randint(0, len(pool) - 1)
            tex = cv2.resize(pool[idx], (w, h))
            return mask, tex

        pairs = [
            _pick("table", table_tex_idx),
            _pick("floor", floor_tex_idx),
            _pick("wall", wall_tex_idx),
        ]
        regions = [p for p in pairs if p is not None]

        # Guarantee at least one region
        if not regions:
            for region, fixed_idx in [("table", table_tex_idx), ("floor", floor_tex_idx), ("wall", wall_tex_idx)]:
                pool = self._textures[region]
                mask = masks.get(region)
                if pool and mask is not None:
                    idx = fixed_idx if fixed_idx is not None else random.randint(0, len(pool) - 1)
                    regions.append((mask, cv2.resize(pool[idx], (w, h))))
                    break

        if not regions:
            return result

        for i in range(len(result)):
            if mode == "paste":
                for mask_seq, tex in regions:
                    m = mask_seq[i] == 255
                    if m.any():
                        result[i][m] = tex[m]
            else:  # hsv
                frame_hsv = cv2.cvtColor(result[i], cv2.COLOR_RGB2HSV)
                for mask_seq, tex in regions:
                    m = mask_seq[i] == 255
                    if m.any():
                        tex_hsv = cv2.cvtColor(tex, cv2.COLOR_RGB2HSV)
                        frame_hsv[m, 0] = tex_hsv[m, 0]
                        frame_hsv[m, 1] = tex_hsv[m, 1]
                result[i] = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2RGB)

        return result
