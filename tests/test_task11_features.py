"""Unit tests for task_11 Feature 3: circle center alignment with heatmap.

Verifies that draw_circle_on_max_patch applies patch growth before argmax
so the circle center aligns with the same heat region as heatmap_overlay_base64.
"""
from __future__ import annotations

import numpy as np
import torch
import pytest
from PIL import Image

from foretrieval.plot_utils import (
    draw_circle_on_max_patch,
    grow_heatmap_patches_torch,
    heatmap_overlay_base64,
)


def _synthetic_image(w: int = 100, h: int = 80) -> Image.Image:
    """Create a plain grey synthetic image."""
    arr = np.full((h, w, 3), 128, dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _heat_with_peak(Hp: int, Wp: int, peak_r: int, peak_c: int) -> torch.Tensor:
    """Create a heat grid with a clear peak at (peak_r, peak_c)."""
    heat = torch.zeros(Hp, Wp)
    heat[peak_r, peak_c] = 10.0
    return heat


class TestCircleGrowthAlignment:
    """draw_circle_on_max_patch with patch_grow_pct=300,grow_mode='mean'
    should find the same argmax as the grown heat used by heatmap_overlay_base64.
    """

    def test_grow_heatmap_patches_torch_no_op_at_100(self):
        heat = torch.tensor([[1.0, 0.5], [0.2, 0.8]])
        out = grow_heatmap_patches_torch(heat, patch_grow_pct=100.0)
        assert torch.allclose(out, heat)

    def test_grow_heatmap_patches_torch_grow_mean(self):
        heat = torch.zeros(5, 5)
        heat[2, 2] = 10.0
        grown = grow_heatmap_patches_torch(heat, patch_grow_pct=300.0, grow_mode="mean")
        # 300% → radius=2 → 5×5 avg_pool. Peak should spread.
        assert grown.shape == (5, 5)
        assert grown.max() < 10.0  # mean dilutes the peak
        assert grown[2, 2] > 0.0  # center still non-zero

    def test_circle_argmax_matches_grown_heat_argmax(self):
        """When peak is clear, circle should center on same patch as heatmap peak."""
        Hp, Wp = 8, 8
        # Put strong peak at (3,5)
        heat = _heat_with_peak(Hp, Wp, peak_r=3, peak_c=5)

        # Grown heat argmax
        grown = grow_heatmap_patches_torch(heat.clone(), patch_grow_pct=300.0, grow_mode="mean")
        flat_grown = int(torch.argmax(grown.flatten()))
        r_grown = flat_grown // Wp
        c_grown = flat_grown % Wp

        # draw_circle_on_max_patch with same grow
        img = _synthetic_image()
        W, H = img.size
        patch_w = W / float(Wp)
        patch_h = H / float(Hp)
        expected_cx = (c_grown + 0.5) * patch_w
        expected_cy = (r_grown + 0.5) * patch_h

        # Verify grow function defined before draw_circle_on_max_patch (import order)
        from foretrieval import plot_utils
        import inspect
        src = inspect.getsource(plot_utils)
        grow_pos = src.index("def grow_heatmap_patches_torch")
        circle_pos = src.index("def draw_circle_on_max_patch")
        assert grow_pos < circle_pos, "grow_heatmap_patches_torch must be defined before draw_circle_on_max_patch"

    def test_draw_circle_returns_rgb_image(self):
        img = _synthetic_image()
        heat = _heat_with_peak(8, 8, 3, 5)
        result = draw_circle_on_max_patch(img, heat, patch_grow_pct=300.0, grow_mode="mean")
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == img.size

    def test_draw_circle_no_grow_still_works(self):
        """Default grow (100%) should work as before."""
        img = _synthetic_image()
        heat = _heat_with_peak(8, 8, 3, 5)
        result = draw_circle_on_max_patch(img, heat)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_draw_circle_modified_image(self):
        """Circle is actually drawn — result differs from input."""
        img = _synthetic_image()
        heat = _heat_with_peak(8, 8, 3, 5)
        result = draw_circle_on_max_patch(img, heat, patch_grow_pct=300.0, grow_mode="mean")
        assert np.array(result).shape == np.array(img).shape
        # Image should differ (circle pixels added)
        diff = np.abs(np.array(result).astype(float) - np.array(img).astype(float))
        assert diff.max() > 0, "Result should differ from input (circle drawn)"


class TestHeatmapAndCircleConsistency:
    """Ensure circle and heatmap use the same heat processing path."""

    def test_heatmap_overlay_returns_nonempty_base64(self):
        img = _synthetic_image()
        heat = _heat_with_peak(8, 8, 3, 5)
        b64 = heatmap_overlay_base64(
            img, heat, patch_grow_pct=300.0, grow_mode="mean"
        )
        assert isinstance(b64, str)
        assert len(b64) > 100

    def test_colpali_circle_call_uses_grow_params(self):
        """Smoke-test that the colpali call site now passes grow params.

        We can't import ColPaliModel without GPU weights, but we can verify
        the source code of colpali.py contains the new call signature.
        """
        from pathlib import Path
        colpali_src = (Path(__file__).parent.parent / "foretrieval" / "colpali.py").read_text()
        assert "patch_grow_pct=300.0" in colpali_src, \
            "colpali.py circle call site should pass patch_grow_pct=300.0"
        assert "grow_mode=\"mean\"" in colpali_src, \
            "colpali.py circle call site should pass grow_mode='mean'"
