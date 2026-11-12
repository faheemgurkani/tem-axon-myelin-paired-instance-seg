import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("nnunetv2")
_spec = importlib.util.spec_from_file_location(
    "tem_trainers", Path(__file__).resolve().parents[1] / "baselines/nnunet/trainers.py")
trainers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trainers)


def test_improvement_resets_patience():
    assert trainers.epochs_since_improvement([0.1, 0.2, 0.3], 1e-3) == (2, 0, 0.3)


def test_gains_below_min_delta_do_not_count():
    best_epoch, stale, best = trainers.epochs_since_improvement([0.5, 0.5005, 0.5009, 0.4], 1e-3)
    assert (best_epoch, stale, best) == (0, 3, 0.5)


def test_plateau_triggers_stop():
    ema = [i / 100 for i in range(50)] + [0.49] * 40
    best_epoch, stale, _ = trainers.epochs_since_improvement(ema, 1e-3)
    assert best_epoch == 49 and stale == 40


class _Logger:
    def __init__(self, ema):
        self.ema = ema

    def get_value(self, key, step):
        assert key == "ema_fg_dice" and step is None
        return self.ema


def _trainer(ema):
    t = trainers.nnUNetTrainer_250epochs_ckpt10_es.__new__(trainers.nnUNetTrainer_250epochs_ckpt10_es)
    t.logger, t.current_epoch, t.print_to_log_file = _Logger(ema), len(ema), lambda *a: None
    return t


def test_should_stop_respects_min_epochs_and_patience():
    assert not _trainer([0.5] * 40)._should_stop()           # flat, but before min_epochs (60)
    assert _trainer([0.5] * 61)._should_stop()               # flat for 60 epochs ≥ patience 30
    assert not _trainer([i / 100 for i in range(100)])._should_stop()  # still improving
