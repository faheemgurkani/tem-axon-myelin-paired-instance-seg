"""nnU-Net trainer variants for preemptible cloud GPUs (Modal).

Identical to ``nnUNetTrainer_250epochs`` except that ``checkpoint_latest`` is written every 10 epochs
instead of every 50. Modal GPU containers can be preempted; with ``--c`` the run resumes from the latest
checkpoint, so this caps the work lost per preemption at 10 epochs. Nothing else changes (D-10).

nnU-Net only discovers trainers inside its own package, so the Modal image copies this file into
``nnunetv2/training/nnUNetTrainer/variants/`` (see ``infra/modal/tem_modal.py``).
"""

import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_250epochs_ckpt10(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):  # noqa: B008 (nnU-Net's signature)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 10


class nnUNetTrainer_5epochs_ckpt1(nnUNetTrainer):
    """Smoke test: 5 epochs, checkpoint every epoch. Used to measure seconds per epoch (D-10)."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):  # noqa: B008 (nnU-Net's signature)
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 5
        self.save_every = 1
