#!/usr/bin/env python3
"""
Backdoor Lottery Hypothesis: Which neurons carry the backdoor?

Hypothesis: Backdoors are encoded in specific subnetworks. By iteratively pruning
neurons, we can identify the "lottery ticket" that carries the backdoor.

Expected outcome:
- BadNets: Ticket is concentrated (few neurons)
- Blended: Ticket is distributed (many neurons)
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from preact_resnet import PreActResNet18
from attacks import BadNetsTrigger, BlendedTrigger


class BackdoorLotteryAnalyzer:
    """Identify backdoor lottery tickets through iterative pruning."""

    def __init__(self, device='cpu'):
        self.device = device

    def prune_magnitude(self, model, prune_ratio):
        """Prune bottom X% of weights by magnitude."""
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                weights = param.data.abs().flatten()
                threshold = torch.quantile(weights, prune_ratio)
                mask = param.data.abs() >= threshold
                param.data = param.data * mask.float()

    def evaluate_model(self, model, clean_loader, triggered_loader, target_class=0):
        """Evaluate clean accuracy and attack success rate."""
        model.eval()

        clean_correct = 0
        clean_total = 0
        with torch.no_grad():
            for images, labels in clean_loader:
                outputs = model(images.to(self.device))
                predictions = torch.argmax(outputs, dim=1)
                clean_correct += (predictions == labels.to(self.device)).sum().item()
                clean_total += labels.size(0)

        clean_acc = clean_correct / clean_total if clean_total > 0 else 0

        asr_correct = 0
        asr_total = 0
        with torch.no_grad():
            for images, _ in triggered_loader:
                outputs = model(images.to(self.device))
                predictions = torch.argmax(outputs, dim=1)
                asr_correct += (predictions == target_class).sum().item()
                asr_total += images.size(0)

        asr = asr_correct / asr_total if asr_total > 0 else 0

        return clean_acc, asr

    def run_lottery_analysis(self, model, clean_loader, triggered_loader, model_name):
        """Run iterative pruning to find backdoor lottery ticket."""
        print(f"\n{'='*70}")
        print(f"BACKDOOR LOTTERY ANALYSIS: {model_name.upper()}")
        print(f"{'='*70}\n")

        results = {
            'model_name': model_name,
            'pruning_steps': [],
        }

        print(f"{'Prune %':<12} {'Clean Acc':<15} {'ASR':<15} {'Gap':<15}")
        print("-" * 60)

        for prune_pct in range(0, 95, 5):
            model_copy = PreActResNet18().to(self.device)
            model_copy.load_state_dict(model.state_dict())

            prune_ratio = prune_pct / 100.0
            self.prune_magnitude(model_copy, prune_ratio)

            clean_acc, asr = self.evaluate_model(model_copy, clean_loader, triggered_loader)
            gap = clean_acc - asr

            print(f"{prune_pct:<12}% {clean_acc:<15.2%} {asr:<15.2%} {gap:<15.2%}")

            results['pruning_steps'].append({
                'prune_pct': prune_pct,
                'clean_acc': float(clean_acc),
                'asr': float(asr),
                'gap': float(gap),
            })

        ticket_size = self._find_ticket_size(results['pruning_steps'])

        print(f"\n{'Backdoor Lottery Ticket Size:':<35} {ticket_size}%\n")

        results['ticket_size'] = ticket_size
        return results

    def _find_ticket_size(self, steps):
        """Find where ASR drops significantly."""
        if len(steps) < 2:
            return 0

        max_drop = 0
        ticket_size = 0

        for i in range(1, len(steps)):
            asr_drop = steps[i-1]['asr'] - steps[i]['asr']
            if asr_drop > max_drop:
                max_drop = asr_drop
                ticket_size = steps[i]['prune_pct']

        return ticket_size


def load_model(checkpoint_path, device):
    """Load trained model."""
    model = PreActResNet18().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    return model


def load_cifar10_loader(num_samples=100, batch_size=32):
    """Load CIFAR-10 data."""
    data_path = Path('./data/cifar-10-batches-py/test_batch')
    if not data_path.exists():
        raise FileNotFoundError(f"CIFAR-10 data not found at {data_path}")

    with open(data_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')

    images = torch.FloatTensor(batch[b'data'][:num_samples]) / 255.0
    images = images.view(-1, 3, 32, 32)
    labels = torch.LongTensor(batch[b'labels'][:num_samples])

    dataset = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def create_triggered_loader(normal_loader, trigger, target_class=0, batch_size=32):
    """Create loader with triggered images."""
    triggered_images = []
    for images, _ in normal_loader:
        triggered = torch.stack([trigger.apply(img) for img in images])
        triggered_images.append(triggered)

    triggered_images = torch.cat(triggered_images, dim=0)
    dummy_labels = torch.full((triggered_images.size(0),), target_class)
    dataset = torch.utils.data.TensorDataset(triggered_images, dummy_labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)


def main():
    print("BACKDOOR LOTTERY HYPOTHESIS ANALYSIS")
    print("="*70)
    print("""
Question: Are BadNets and Blended backdoors encoded in different-sized
neural network subsets (lottery tickets)?

Hypothesis:
- BadNets: Concentrated backdoor -> Small lottery ticket
- Blended: Distributed backdoor -> Large lottery ticket

Method: Iterative magnitude pruning
Measure: How quickly ASR drops as we prune
""")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    try:
        print("Loading CIFAR-10...")
        clean_loader = load_cifar10_loader(num_samples=200, batch_size=32)

        print("Loading trained models...")
        badnets_model = load_model('./checkpoints/badnets.pth', device)
        blended_model = load_model('./checkpoints/blended.pth', device)

        print("Creating triggers...")
        badnets_trigger = BadNetsTrigger(patch_size=3, image_size=32)
        blended_trigger = BlendedTrigger(image_size=32, alpha=0.15, seed=0)

        badnets_triggered = create_triggered_loader(clean_loader, badnets_trigger, batch_size=32)
        blended_triggered = create_triggered_loader(clean_loader, blended_trigger, batch_size=32)

        analyzer = BackdoorLotteryAnalyzer(device)

        print("\n" + "="*70)
        badnets_results = analyzer.run_lottery_analysis(badnets_model, clean_loader, badnets_triggered, 'BadNets')

        print("\n" + "="*70)
        blended_results = analyzer.run_lottery_analysis(blended_model, clean_loader, blended_triggered, 'Blended')

        print("\n" + "="*70)
        print("LOTTERY TICKET COMPARISON")
        print("="*70)

        badnets_ticket = badnets_results['ticket_size']
        blended_ticket = blended_results['ticket_size']

        print(f"\nBadNets ticket size:  {badnets_ticket}%")
        print(f"Blended ticket size:  {blended_ticket}%")

        if blended_ticket > badnets_ticket:
            print(f"\n✓ HYPOTHESIS SUPPORTED")
            print(f"  Blended backdoor is encoded in MORE neurons")
            verdict = "SUPPORTED"
        else:
            print(f"\n✗ HYPOTHESIS NOT SUPPORTED")
            verdict = "NOT SUPPORTED"

        results_file = Path('./results/lottery_hypothesis_results.json')
        results_file.parent.mkdir(exist_ok=True, parents=True)

        all_results = {
            'badnets': badnets_results,
            'blended': blended_results,
            'comparison': {
                'badnets_ticket': badnets_ticket,
                'blended_ticket': blended_ticket,
                'verdict': verdict,
            }
        }

        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\n✓ Results saved to {results_file}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
