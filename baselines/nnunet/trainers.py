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
