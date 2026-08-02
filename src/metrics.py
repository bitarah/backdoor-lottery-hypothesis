"""ARR / ASR / RDR metrics, exact definitions from Appendix C of Dunnett et al. (2025)."""
import torch


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


@torch.no_grad()
def asr(model, asr_loader, device):
    """Attack success rate: accuracy of predicting the (fixed) target label on triggered,
    non-target-class test samples. asr_loader must already yield (triggered_x, target_label)."""
    return accuracy(model, asr_loader, device)


def arr(clean_acc_before, clean_acc_after):
    """Accuracy Reduction Ratio, Eq. 8: ARR = 1 - a_s / a_p."""
    return 1 - (clean_acc_after / clean_acc_before)


def rdr(recovery_acc_after, clean_acc_before):
    """Recovery Difference Ratio, Eq. 9: RDR = 1 - eta_s / a_p, where eta_s is the accuracy of
    the DEFENDED model on triggered inputs evaluated against their ORIGINAL (clean-task)
    labels (i.e. did the defense restore the correct, non-backdoor prediction?)."""
    return 1 - (recovery_acc_after / clean_acc_before)
