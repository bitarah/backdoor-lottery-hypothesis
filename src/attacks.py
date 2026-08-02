"""Backdoor attack implementations, matching the threat model in Dunnett et al. (2025),
"Backdoor Mitigation via Invertible Pruning Masks" (arXiv:2509.15497), Sec. 3 ("Threat Model"):
a targeted attack where the adversary trains the model on a clean dataset D_c plus a poisoned
dataset D_b = {(x-hat, t)} of trigger-stamped inputs labelled with a fixed target class t.

Two representative attacks from the paper's 8-attack suite (Sec 5.1) are implemented here:
BadNets (Gu et al. 2017) -- the seminal, simplest patch trigger -- and Blended (Chen et al.
2017) -- an alpha-blended perturbation trigger, representing the "patch" and "blended noise"
attack families respectively.
"""
import numpy as np
import torch


class BadNetsTrigger:
    """A 3x3 white square in the bottom-right corner, as in the original BadNets paper."""

    def __init__(self, patch_size=3, image_size=32):
        self.patch_size = patch_size
        self.image_size = image_size

    def apply(self, img_tensor):
        # img_tensor: (C, H, W) float in [0, 1]
        img = img_tensor.clone()
        s = self.patch_size
        img[:, -s:, -s:] = 1.0
        return img


class BlendedTrigger:
    """Alpha-blends a fixed random pattern into the image (Chen et al. 2017)."""

    def __init__(self, image_size=32, alpha=0.15, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.pattern = torch.rand(3, image_size, image_size, generator=g)
        self.alpha = alpha

    def apply(self, img_tensor):
        return (1 - self.alpha) * img_tensor + self.alpha * self.pattern


ATTACKS = {"badnets": BadNetsTrigger, "blended": BlendedTrigger}


class PoisonedDataset(torch.utils.data.Dataset):
    """Wraps a clean image dataset, poisoning a subset with a trigger + target label.

    mode='train': poison a `poison_ratio` fraction of samples (label -> target_label).
    mode='test_asr': return ONLY non-target-class samples, all triggered, all labelled
                      target_label (used to compute Attack Success Rate).
    mode='clean': no poisoning at all (pass-through), used for clean accuracy / RDR.
    """

    def __init__(self, base_dataset, attack_name, target_label=0, poison_ratio=0.1,
                 mode="train", seed=0):
        self.base = base_dataset
        self.trigger = ATTACKS[attack_name](image_size=32)
        self.target_label = target_label
        self.mode = mode
        rng = np.random.default_rng(seed)

        if mode == "train":
            n = len(base_dataset)
            n_poison = int(n * poison_ratio)
            self.poison_idx = set(rng.choice(n, size=n_poison, replace=False).tolist())
            self.indices = list(range(n))
        elif mode == "test_asr":
            self.indices = [i for i in range(len(base_dataset)) if base_dataset[i][1] != target_label]
            self.poison_idx = set(self.indices)
        elif mode == "clean":
            self.indices = list(range(len(base_dataset)))
            self.poison_idx = set()
        else:
            raise ValueError(mode)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        img, label = self.base[idx]
        if idx in self.poison_idx:
            img = self.trigger.apply(img)
            label = self.target_label
        return img, label
