"""Defenses: FP (Fine-Pruning, Liu et al. 2018 -- the classic baseline the paper positions
itself against) and IMS (Invertible Masking using Selection, Dunnett et al. 2025), reimplemented
directly from the paper's equations (Sec 4, Eqs 1-7) rather than the entangled BackdoorBench
framework, using hooks on every Conv2d layer of the (real) PreActResNet18.
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_conv_modules(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


# ---------------------------------------------------------------------------
# FP: Fine-Pruning (Liu, Dolan-Gavitt & Garg, 2018) -- classic baseline
# ---------------------------------------------------------------------------

def fine_pruning(model, clean_loader, device, prune_frac=0.6, finetune_epochs=5, lr=0.01):
    """Prune the channels of the LAST conv layer with lowest mean clean activation, then
    fine-tune the remaining network on clean data (the two-step FP recipe)."""
    model = copy.deepcopy(model).to(device)
    last_conv = get_conv_modules(model)[-1]

    activations = []

    def hook(module, inp, out):
        activations.append(out.detach().mean(dim=(0, 2, 3)))

    h = last_conv.register_forward_hook(hook)
    model.eval()
    with torch.no_grad():
        for x, _ in clean_loader:
            model(x.to(device))
    h.remove()
    mean_act = torch.stack(activations).mean(0)
    n_prune = int(len(mean_act) * prune_frac)
    prune_idx = torch.argsort(mean_act)[:n_prune]

    mask = torch.ones(last_conv.out_channels, device=device)
    mask[prune_idx] = 0.0

    def prune_hook(module, inp, out):
        return out * mask.view(1, -1, 1, 1)

    last_conv.register_forward_hook(prune_hook)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    model.train()
    for _ in range(finetune_epochs):
        for x, y in clean_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
    return model


# ---------------------------------------------------------------------------
# IMS: Invertible Masking using Selection (Dunnett et al. 2025)
# ---------------------------------------------------------------------------

class MaskHook:
    """Holds the per-layer channel mask logit `a` and selection logit `s`, and applies
    a' = sigmoid(k(a-0.5)) + s*(1-sigmoid(k(a-0.5)))         [Eq. 1, forward mask]
    a-bar' = sigmoid(k((1-a)-0.5)) + s*(1-sigmoid(k((1-a)-0.5)))  [Eq. 2, inverse mask]
    to the output feature map of a Conv2d layer, matching the official ims.py hook design."""

    def __init__(self, out_channels, k=20.0, device="cpu"):
        self.a = torch.zeros(out_channels, device=device, requires_grad=True)  # raw mask param
        self.s = torch.zeros(out_channels, device=device, requires_grad=True)  # raw selection param
        self.k = k
        self.use_inverse = False
        self.enabled = True

    def _mask_value(self, a_raw):
        sig_a = torch.sigmoid(self.k * (a_raw - 0.5))
        s = torch.sigmoid(self.s)  # keep selection in [0,1]
        return sig_a + s * (1 - sig_a)

    def hook_fn(self, module, inp, out):
        if not self.enabled:
            return out
        a_eff = (1 - torch.sigmoid(self.a)) if self.use_inverse else torch.sigmoid(self.a)
        mask = self._mask_value(a_eff)
        return out * mask.view(1, -1, 1, 1)

    def params(self):
        return [self.a, self.s]


def setup_hooks(model, device):
    hooks, handles = [], []
    for conv in get_conv_modules(model):
        h = MaskHook(conv.out_channels, device=device)
        handle = conv.register_forward_hook(h.hook_fn)
        hooks.append(h)
        handles.append(handle)
    return hooks, handles


def set_inverse(hooks, inverse: bool):
    for h in hooks:
        h.use_inverse = inverse


def set_enabled(hooks, enabled: bool):
    for h in hooks:
        h.enabled = enabled


def agree_loss(logits1, logits2):
    p1, p2 = F.softmax(logits1, 1), F.softmax(logits2, 1)
    overlap = (p1 * p2).sum(1)
    return -torch.log(overlap + 1e-8).mean()


def disagree_loss(logits1, logits2):
    p1, p2 = F.softmax(logits1, 1), F.softmax(logits2, 1)
    overlap = (p1 * p2).sum(1)
    return -torch.log(1.0 - overlap + 1e-8).mean()


def sparsity_penalty(hooks):
    s = torch.cat([torch.sigmoid(h.s) for h in hooks])
    return s.mean()


def ims_defense(model, clean_loader, device, k=20.0,
                 init_steps=200, outer_steps=200, inner_steps=5,
                 eps=1.0, lr_mask=1e-2, lr_pert=0.5, lambda_final=10.0):
    """Reimplements Sec 4.2's bi-level optimisation:
      1) Mask Initialisation (Eq. 4): find (a,s) s.t. the forward mask preserves clean
         predictions while the inverse mask corrupts them (isolating clean-only components).
      2) Inner subproblem (Eq. 5): synthesize a bounded perturbation delta per batch that the
         INVERSE-masked model still classifies like the reference model, but that changes the
         reference model's own prediction -- a proxy backdoor trigger built only from clean data.
      3) Outer subproblem (Eqs. 6-7): update (a,s) to neutralise that perturbation's effect
         under the forward mask, while keeping the Eq. 4 objectives satisfied.
    `model` (the backdoored network) is frozen throughout; only mask logits (a,s) are learned.
    Operates on a deep copy so the caller's original model/hooks are never mutated.
    """
    model = copy.deepcopy(model).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    hooks, handles = setup_hooks(model, device)
    mask_params = [p for h in hooks for p in h.params()]
    for p in mask_params:
        p.requires_grad_(True)
    mask_opt = torch.optim.AdamW(mask_params, lr=lr_mask)

    def forward_masked(x, inverse):
        set_enabled(hooks, True)
        set_inverse(hooks, inverse)
        return model(x)

    def forward_ref(x):
        set_enabled(hooks, False)
        return model(x)

    # ---- 1) Mask initialisation (Eq. 4) ----
    it = iter(clean_loader)
    for step in range(init_steps):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(clean_loader)
            x, _ = next(it)
        x = x.to(device)
        pred_ref = forward_ref(x).detach()
        pred_masked = forward_masked(x, inverse=False)
        pred_inv = forward_masked(x, inverse=True)
        loss = agree_loss(pred_masked, pred_ref) + disagree_loss(pred_inv, pred_ref) \
            + 0.1 * sparsity_penalty(hooks)
        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()

    # ---- 2)/3) Inner + outer subproblems (Eqs. 5-7) ----
    lambdas = [0.0] * (outer_steps // 2) + [lambda_final] * (outer_steps - outer_steps // 2)
    it = iter(clean_loader)
    for step in range(outer_steps):
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(clean_loader)
            x, _ = next(it)
        x = x.to(device)

        # --- inner: synthesize perturbation delta ---
        delta = torch.zeros_like(x, requires_grad=True)
        pert_opt = torch.optim.AdamW([delta], lr=lr_pert)
        pred_ref_clean = forward_ref(x).detach()
        for _ in range(inner_steps):
            x_pert = (x + delta).clamp(0, 1)
            pred_ref_pert = forward_ref(x_pert)
            pred_inv_pert = forward_masked(x_pert, inverse=True)
            loss_inner = disagree_loss(pred_ref_pert, pred_ref_clean) \
                + agree_loss(pred_ref_pert, pred_inv_pert)
            pert_opt.zero_grad()
            loss_inner.backward()
            pert_opt.step()
            delta.data.clamp_(-eps, eps)
        delta = delta.detach()
        x_pert = (x + delta).clamp(0, 1)

        # --- outer: update mask to neutralise delta while keeping Eq.4 objectives ---
        pred_masked_clean = forward_masked(x, inverse=False)
        pred_masked_pert = forward_masked(x_pert, inverse=False)
        pred_inv_clean = forward_masked(x, inverse=True)
        pred_inv_pert = forward_masked(x_pert, inverse=True)
        pred_ref_pert = forward_ref(x_pert).detach()

        loss_outer = (
            agree_loss(pred_masked_clean, pred_ref_clean)
            + agree_loss(pred_masked_pert, pred_ref_clean)
            + disagree_loss(pred_inv_pert, pred_ref_clean)
            + agree_loss(pred_ref_pert, pred_inv_pert)
            + disagree_loss(pred_inv_clean, pred_ref_clean)
        )
        loss_outer = loss_outer + lambdas[step] * sparsity_penalty(hooks)

        mask_opt.zero_grad()
        loss_outer.backward()
        mask_opt.step()

    # Freeze final mask in "forward" (backdoor-suppressing) mode for downstream evaluation.
    set_enabled(hooks, True)
    set_inverse(hooks, False)
    for h in hooks:
        h.a.requires_grad_(False)
        h.s.requires_grad_(False)
    return model, hooks
