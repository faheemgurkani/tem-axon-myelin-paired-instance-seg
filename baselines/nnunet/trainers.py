"""nnU-Net trainer variants for preemptible, pay-per-second cloud GPUs (Modal).

nnU-Net only discovers trainers inside its own package, so the Modal image copies this file into
``nnunetv2/training/nnUNetTrainer/variants/tem/`` (see ``infra/modal/tem_modal.py``).

- ``nnUNetTrainer_250epochs_ckpt10``: ``nnUNetTrainer_250epochs`` but ``checkpoint_latest`` every 10 epochs
  instead of 50, so a preemption loses at most 10 epochs when the run resumes with ``--c`` (D-10).
- ``nnUNetTrainer_250epochs_ckpt10_es``: the same, plus early stopping on nnU-Net's EMA pseudo-Dice. This is
  the default on Modal: GPU seconds stop being billed once validation stops improving.
- ``nnUNetTrainer_5epochs_ckpt1``: a short timing run to measure seconds per epoch before a long run.
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


def epochs_since_improvement(ema: list, min_delta: float) -> tuple[int, int, float]:
    """Return (best_epoch, epochs since it, best value). Only a gain above ``min_delta`` counts as improvement."""
    best, best_epoch = float("-inf"), 0
    for epoch, value in enumerate(ema):
        if value > best + min_delta:
            best, best_epoch = value, epoch
    return best_epoch, len(ema) - 1 - best_epoch, best


class nnUNetTrainer_250epochs_ckpt10(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):  # noqa: B008 (nnU-Net's signature)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 10


class nnUNetTrainer_250epochs_ckpt10_es(nnUNetTrainer_250epochs_ckpt10):
    """Early stopping on the EMA pseudo-Dice (``ema_fg_dice``), the signal nnU-Net already uses for checkpoint_best.

    The stopping rule:
    - Training stops once the EMA has not improved by more than ``min_delta`` for ``patience`` epochs.
    - It never stops before ``min_epochs``, so the poly learning rate has time to work.
    - 250 epochs remains the hard cap.

    The rule reads the logged history, so it still holds after a resume with ``--c`` (the logger is restored from
    the checkpoint). Run the final validation with ``--val_best`` so it uses ``checkpoint_best``, not the weights
    from ``patience`` epochs later.

    Caveat for the write-up: nnU-Net's poly schedule is defined over ``num_epochs``. A run stopped early ends at a
    higher learning rate than a full run, which is why ``--val_best`` is used.
    """

    patience = 30
    min_epochs = 60
    min_delta = 1e-3

    def _should_stop(self) -> bool:
        ema = list(self.logger.get_value('ema_fg_dice', step=None))
        if len(ema) < self.min_epochs:
            return False
        best_epoch, stale, best = epochs_since_improvement(ema, self.min_delta)
        if stale >= self.patience:
            self.print_to_log_file(
                f"Early stopping at epoch {self.current_epoch - 1}: EMA pseudo Dice has not improved by "
                f"> {self.min_delta} for {stale} epochs (best {best:.4f} at epoch {best_epoch})."
            )
            return True
        return False

    def run_training(self):
        # Same loop as nnUNetTrainer.run_training, with a stop check at each epoch boundary.
        self.on_train_start()
        while self.current_epoch < self.num_epochs and not self._should_stop():
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = []
            for _ in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for _ in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
        self.on_train_end()


class nnUNetTrainer_150epochs_ckpt10_es(nnUNetTrainer_250epochs_ckpt10_es):
    """B1 on a $10 budget (D-10 revised, 13 Nov). The timing run measured 142 s per epoch on an L4 with the
    2d_bigpatch configuration. So 250 epochs ≈ 9.9 GPU-h ≈ $12 does not fit, while 150 epochs ≈ 5.9 GPU-h does.

    The schedule length is cut instead of capping wall-time, so that nnU-Net's poly learning rate still decays
    to ~0 at the end. Early stopping (patience 30, min 60 epochs) still applies.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):  # noqa: B008 (nnU-Net's signature)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150


class nnUNetTrainer_150epochs_ckpt10(nnUNetTrainer_250epochs_ckpt10):
    """B1's recipe at a FIXED 150 epochs, no early stopping (D-39, 16 Nov). For the TEM1 reference (T-22/T-24):
    its nnU-Net validation fold is the held-out test mice, so early stopping would pick the stop epoch on the test
    set. A fixed schedule makes no decision on them; predictions use checkpoint_final."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):  # noqa: B008 (nnU-Net's signature)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150


class nnUNetTrainer_5epochs_ckpt1(nnUNetTrainer):
    """Timing run: 5 epochs, checkpoint every epoch. Used to measure seconds per epoch (D-10)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):  # noqa: B008 (nnU-Net's signature)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
        self.save_every = 1

    def perform_actual_validation(self, save_probabilities: bool = False):
        # A 5-epoch model's full-image validation is meaningless and costs ~10–20 GPU-minutes; skip it.
        self.print_to_log_file("Timing run: final validation skipped.")


class nnUNetTrainer_80epochs_ckpt10(nnUNetTrainer_250epochs_ckpt10):
    """T-24 arms R1–R3 (C3 pilot, D-35): B1's recipe at a fixed 80 epochs, no early stopping (the validation
    fold is the held-out test mice, as for the reference, D-39). Predictions use checkpoint_final."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):  # noqa: B008 (nnU-Net's signature)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 80


WEIGHT_LEVELS = 20  # weight channel stores d in 0..20; pixel weight = 1 - d / 20 (padding -1 -> 0 -> weight 1)


class SoftWeightDCCE(torch.nn.Module):
    """Dice + CE with a per-pixel weight from a second target channel (T-24 R4, D-35/D-42).

    ``target``: (b, 1, ...) labels, or (b, 2, ...) labels + down-weight level d (0..WEIGHT_LEVELS). The weight
    1 − d/WEIGHT_LEVELS multiplies the Dice ``loss_mask`` and the per-pixel CE; ignore-label pixels get weight 0.
    With d ≡ 0 this equals nnU-Net's ``DC_and_CE_loss`` (CE is a weighted mean over the same pixels).
    d is stored as a *down-weight* so that the spatial transform's −1 padding (set to 0 by RemoveLabelTansform)
    means full weight, exactly as padding is treated in the other arms.
    """

    def __init__(self, dc: torch.nn.Module, ignore_label: int | None):
        super().__init__()
        self.dc, self.ignore_label = dc, ignore_label

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        seg = target[:, :1]
        if target.shape[1] > 1:
            w = 1.0 - target[:, 1:2].float().clamp(0, WEIGHT_LEVELS) / WEIGHT_LEVELS
        else:
            w = torch.ones_like(seg, dtype=torch.float32)
        if self.ignore_label is not None:
            valid = seg != self.ignore_label
            w = w * valid
            seg = torch.where(valid, seg, torch.zeros_like(seg))
        dc = self.dc(net_output, seg, loss_mask=w)
        ce_pix = torch.nn.functional.cross_entropy(net_output, seg[:, 0].long(), reduction="none")
        den = w.sum()
        ce = (ce_pix * w[:, 0]).sum() / den if den > 0 else ce_pix.sum() * 0
        return ce + dc


class nnUNetTrainer_80epochs_ckpt10_softw(nnUNetTrainer_80epochs_ckpt10):
    """T-24 arm R4: selection-gated soft ignore. The preprocessed seg carries a 2nd channel (down-weight level,
    see ``SoftWeightDCCE``), built by ``scripts/run/t24_build_arms.py``. Validation strips it (unweighted loss and
    pseudo-Dice, like the other arms)."""

    def _build_loss(self):
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.utilities.helpers import softmax_helper_dim1

        assert not self.label_manager.has_regions
        dc = MemoryEfficientSoftDiceLoss(apply_nonlin=softmax_helper_dim1,
                                         batch_dice=self.configuration_manager.batch_dice,
                                         do_bg=False, smooth=1e-5, ddp=self.is_ddp)
        loss = SoftWeightDCCE(dc, self.label_manager.ignore_label)
        if self.enable_deep_supervision:
            scales = self._get_deep_supervision_scales()
            weights = [1 / (2 ** i) for i in range(len(scales))]
            weights[-1] = 0
            s = sum(weights)
            loss = DeepSupervisionWrapper(loss, [x / s for x in weights])
        return loss

    def validation_step(self, batch: dict) -> dict:
        t = batch["target"]
        batch["target"] = [x[:, :1] for x in t] if isinstance(t, list) else t[:, :1]
        return super().validation_step(batch)
