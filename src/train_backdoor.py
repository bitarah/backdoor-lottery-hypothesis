import argparse
import pathlib
import sys
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from preact_resnet import PreActResNet18
from attacks import PoisonedDataset

DATA_ROOT = str(pathlib.Path(__file__).parent.parent / "data")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attack", required=True, choices=["badnets", "blended"])
    p.add_argument("--poison_ratio", type=float, default=0.1)
    p.add_argument("--target_label", type=int, default=0)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = get_device()
    print(f"device={device}")

    to_tensor = T.ToTensor()
    train_base = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=to_tensor)
    test_base = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=to_tensor)

    train_ds = PoisonedDataset(train_base, args.attack, args.target_label, args.poison_ratio, mode="train")
    clean_test_ds = PoisonedDataset(test_base, args.attack, args.target_label, mode="clean")
    asr_test_ds = PoisonedDataset(test_base, args.attack, args.target_label, mode="test_asr")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    clean_loader = DataLoader(clean_test_ds, batch_size=256, shuffle=False, num_workers=0)
    asr_loader = DataLoader(asr_test_ds, batch_size=256, shuffle=False, num_workers=0)

    model = PreActResNet18(num_classes=10).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    # Linear LR warmup over the first epoch: cold-start SGD at lr=0.1 on PreActResNet18
    # produces a transient gradient spike (grad_norm ~130) in the first ~10 batches that
    # corrupts BatchNorm running stats badly enough to diverge to NaN within one epoch.
    warmup_steps = len(train_loader)

    t0 = time.time()
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        tot_loss = 0.0
        for x, y in train_loader:
            if global_step < warmup_steps:
                warmup_lr = args.lr * (global_step + 1) / warmup_steps
                for g in opt.param_groups:
                    g["lr"] = warmup_lr
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss ({loss.item()}) at epoch {epoch}, step {global_step}. "
                    "Training diverged despite warmup + gradient clipping.")
            tot_loss += loss.item() * x.size(0)
            global_step += 1
        if global_step >= warmup_steps:
            sched.step()
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            ca = evaluate(model, clean_loader, device)
            asr = evaluate(model, asr_loader, device)
            print(f"epoch {epoch:3d}  loss={tot_loss/len(train_ds):.4f}  CA={ca:.4f}  ASR={asr:.4f}  "
                  f"({time.time()-t0:.1f}s elapsed)", flush=True)

    ca = evaluate(model, clean_loader, device)
    asr = evaluate(model, asr_loader, device)
    print(f"FINAL  CA={ca:.4f}  ASR={asr:.4f}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "attack": args.attack,
                "poison_ratio": args.poison_ratio, "target_label": args.target_label,
                "clean_acc": ca, "asr": asr}, out_path)
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
