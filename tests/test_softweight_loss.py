"""T-24 R4 soft-weight loss (D-42): equals nnU-Net's DC_and_CE_loss at weight 1; weight-0 pixels get no gradient."""

import importlib.util
from pathlib import Path

import pytest
import torch

pytest.importorskip("nnunetv2")
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss  # noqa: E402
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss  # noqa: E402
from nnunetv2.utilities.helpers import softmax_helper_dim1  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "tem_trainers", Path(__file__).resolve().parents[1] / "baselines/nnunet/trainers.py")
tt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tt)


def _dc():
    return MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, smooth=1e-5,
                                       ddp=False)


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 3, 32, 32, generator=g, requires_grad=True)
    y = torch.randint(0, 4, (2, 1, 32, 32), generator=g)  # 3 = ignore
    return x, y


def test_equals_nnunet_loss_at_full_weight():
    x, y = _batch()
    ref = DC_and_CE_loss({"batch_dice": True, "smooth": 1e-5, "do_bg": False, "ddp": False}, {}, ignore_label=3,
                         dice_class=MemoryEfficientSoftDiceLoss)(x, y)
    ours1 = tt.SoftWeightDCCE(_dc(), 3)(x, y)
    ours2 = tt.SoftWeightDCCE(_dc(), 3)(x, torch.cat([y, torch.zeros_like(y)], 1))
    assert torch.allclose(ref, ours1, atol=1e-6) and torch.allclose(ref, ours2, atol=1e-6)


def test_zero_weight_pixels_get_no_gradient():
    x, y = _batch(1)
    d = torch.zeros_like(y)
    d[:, :, :8] = tt.WEIGHT_LEVELS  # weight 0 on the top 8 rows
    tt.SoftWeightDCCE(_dc(), 3)(x, torch.cat([y, d], 1)).backward()
    assert x.grad[:, :, :8].abs().max() == 0
    assert x.grad[:, :, 8:].abs().max() > 0


def test_partial_weight_scales_ce():
    x, y = _batch(2)
    y = torch.where(y == 3, torch.zeros_like(y), y)
    half = torch.full_like(y, tt.WEIGHT_LEVELS // 2)
    loss_half = tt.SoftWeightDCCE(_dc(), None)(x, torch.cat([y, half], 1))
    loss_full = tt.SoftWeightDCCE(_dc(), None)(x, y)
    # uniform weights cancel in both the weighted-mean CE and the Dice ratios (up to smoothing)
    assert torch.allclose(loss_half, loss_full, atol=1e-4)
