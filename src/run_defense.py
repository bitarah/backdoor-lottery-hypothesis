import argparse
import csv
import pathlib
import sys

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from preact_resnet import PreActResNet18
from attacks import PoisonedDataset
from defenses import fine_pruning, ims_defense, set_enabled
from metrics import accuracy, asr, arr, rdr

DATA_ROOT = str(pathlib.Path(__file__).parent.parent / "data")


def get_device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def spc_subset(dataset, spc, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    by_class = {c: [] for c in range(num_classes)}
    for i in range(len(dataset)):
        by_class[dataset[i][1]].append(i)
    idx = []
    for c in range(num_classes):
        chosen = rng.choice(by_class[c], size=min(spc, len(by_class[c])), replace=False)
        idx.extend(chosen.tolist())
    return Subset(dataset, idx)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--spc", type=int, default=100)
    p.add_argument("--out", required=True)
    p.add_argument("--defenses", nargs="+", default=["fp", "ims"])
    args = p.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    attack, target_label = ckpt["attack"], ckpt["target_label"]

    model = PreActResNet18(num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])

    to_tensor = T.ToTensor()
    train_base = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=to_tensor)
    test_base = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=to_tensor)

    clean_defense_ds = spc_subset(
        PoisonedDataset(train_base, attack, target_label, mode="clean"), args.spc)
    clean_test_ds = PoisonedDataset(test_base, attack, target_label, mode="clean")
    asr_test_ds = PoisonedDataset(test_base, attack, target_label, mode="test_asr")

    clean_defense_loader = DataLoader(clean_defense_ds, batch_size=64, shuffle=True)
    clean_test_loader = DataLoader(clean_test_ds, batch_size=256, shuffle=False)
    asr_test_loader = DataLoader(asr_test_ds, batch_size=256, shuffle=False)

    a_p = accuracy(model, clean_test_loader, device)  # pre-defense clean acc
    asr_p = asr(model, asr_test_loader, device)
    print(f"[{attack}] BEFORE DEFENSE: clean_acc={a_p:.4f}  ASR={asr_p:.4f}")

    # RDR "recovery" accuracy: correct classification of triggered inputs against their
    # ORIGINAL (clean-task) label rather than the attack's target label.
    class TriggeredOriginalLabel(torch.utils.data.Dataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, i):
            idx = self.base.indices[i]
            img, label = self.base.base[idx]
            img = self.base.trigger.apply(img)
            return img, label

    recovery_loader = DataLoader(TriggeredOriginalLabel(
        PoisonedDataset(test_base, attack, target_label, mode="test_asr")), batch_size=256, shuffle=False)

    rows = []
    for defense_name in args.defenses:
        if defense_name == "fp":
            defended = fine_pruning(model, clean_defense_loader, device,
                                     prune_frac=0.6, finetune_epochs=5, lr=0.01)
            a_s = accuracy(defended, clean_test_loader, device)
            asr_s = asr(defended, asr_test_loader, device)
            eta_s = accuracy(defended, recovery_loader, device)
        elif defense_name == "ims":
            defended, hooks = ims_defense(model, clean_defense_loader, device,
                                           init_steps=150, outer_steps=150, inner_steps=5)
            a_s = accuracy(defended, clean_test_loader, device)
            asr_s = asr(defended, asr_test_loader, device)
            eta_s = accuracy(defended, recovery_loader, device)
            set_enabled(hooks, True)
        else:
            raise ValueError(defense_name)

        row = dict(attack=attack, defense=defense_name, clean_acc_before=a_p, asr_before=asr_p,
                   clean_acc_after=a_s, asr_after=asr_s, recovery_acc=eta_s,
                   ARR=arr(a_p, a_s), ASR=asr_s, RDR=rdr(eta_s, a_p))
        print(row)
        rows.append(row)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
