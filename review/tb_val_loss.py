"""Print the logged validation-loss curve minimum of main.py runs (from TensorBoard).

    python -m review.tb_val_loss RUN_DIR [RUN_DIR ...]

main.py logs validation loss with add_scalars("Training vs. Validation Loss", ...),
which TensorBoard stores in the sub-directory "<run>/Training vs. Validation Loss_Validation"
at step epoch+1. These are the numbers that picked model.pt during training, computed
with the random val flips of the original pipeline.
"""
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    print("run_dir\tepochs_logged\tmin_val_loss\tbest_epoch")
    for rd in sys.argv[1:]:
        sub = Path(rd) / "Training vs. Validation Loss_Validation"
        if not sub.is_dir():
            print(f"{rd}\t0\tNA\tNA")
            continue
        ea = EventAccumulator(str(sub))
        ea.Reload()
        tags = ea.Tags().get("scalars", [])
        if not tags:
            print(f"{rd}\t0\tNA\tNA")
            continue
        ev = ea.Scalars(tags[0])
        best = min(ev, key=lambda e: e.value)
        print(f"{rd}\t{len(ev)}\t{best.value:.6f}\t{best.step - 1}")


if __name__ == "__main__":
    main()
