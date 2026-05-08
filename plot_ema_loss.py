#!/usr/bin/env python3
import argparse
import csv
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="matplotlib-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FLOAT_RE = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
EPOCH_PATTERNS = [
    re.compile(r"\bEpoch\s*\[\s*(\d+)\s*/\s*\d+\s*\]", re.IGNORECASE),
    re.compile(r"\bEpoch\s+(\d+)\s*/\s*\d+", re.IGNORECASE),
    re.compile(r"\bEpoch\s+(\d+)\b", re.IGNORECASE),
    re.compile(r"\bepoch\s*[:=]\s*(\d+)\b", re.IGNORECASE),
]
LOSS_PATTERNS = [
    re.compile(r"\btotal[_\s-]*loss\s*[:=]\s*" + FLOAT_RE, re.IGNORECASE),
    re.compile(r"\bfinal[_\s-]*loss\s*[:=]\s*" + FLOAT_RE, re.IGNORECASE),
    re.compile(r"(?<![_A-Za-z])loss\s*[:=]\s*" + FLOAT_RE, re.IGNORECASE),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot EMA loss from train.log beside this script.")
    parser.add_argument("--alpha", default=0.9, type=float, help="EMA alpha, default: 0.9")
    return parser.parse_args()


def extract_epoch(line):
    for pattern in EPOCH_PATTERNS:
        match = pattern.search(line)
        if match:
            return int(match.group(1))
    return None


def extract_loss(line):
    for pattern in LOSS_PATTERNS:
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def parse_train_log(log_path):
    epoch_losses = defaultdict(list)
    current_epoch = None
    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            epoch = extract_epoch(line)
            if epoch is not None:
                current_epoch = epoch
            loss = extract_loss(line)
            if loss is None:
                continue
            target_epoch = epoch if epoch is not None else current_epoch
            if target_epoch is not None:
                epoch_losses[target_epoch].append(loss)
    return epoch_losses


def compute_epoch_level_loss(epoch_losses):
    epochs = sorted(epoch_losses)
    raw_losses = []
    for epoch in epochs:
        losses = epoch_losses[epoch]
        raw_losses.append(sum(losses) / len(losses))
    return epochs, raw_losses


def compute_ema(raw_losses, alpha):
    ema_losses = []
    for loss in raw_losses:
        if not ema_losses:
            ema_losses.append(loss)
        else:
            ema_losses.append(alpha * ema_losses[-1] + (1 - alpha) * loss)
    return ema_losses


def descent_rate(ema_losses, index, window):
    if index < window:
        return ""
    prev = ema_losses[index - window]
    return (prev - ema_losses[index]) / (prev + 1e-12)


def write_csv(csv_path, epochs, raw_losses, ema_losses):
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "raw_loss", "ema_loss", "descent_rate_w3", "descent_rate_w5"])
        for index, epoch in enumerate(epochs):
            w3 = descent_rate(ema_losses, index, 3)
            w5 = descent_rate(ema_losses, index, 5)
            writer.writerow([
                epoch,
                "{:.10f}".format(raw_losses[index]),
                "{:.10f}".format(ema_losses[index]),
                "" if w3 == "" else "{:.10f}".format(w3),
                "" if w5 == "" else "{:.10f}".format(w5),
            ])


def plot_curve(png_path, epochs, raw_losses, ema_losses, alpha):
    plt.figure(figsize=(9, 5.5))
    plt.plot(epochs, raw_losses, linestyle="--", color="#9aa4b2", alpha=0.55, label="Raw epoch loss")
    plt.plot(epochs, ema_losses, color="#1f77b4", linewidth=2.2, label="EMA loss")
    plt.xlabel("Epoch")
    plt.ylabel("EMA loss")
    plt.title("EMA Loss Curve (alpha={:.3g})".format(alpha))
    plt.grid(True, linestyle="--", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=180)
    plt.close()


def main():
    args = parse_args()
    if not 0 <= args.alpha <= 1:
        raise ValueError("--alpha must be in [0, 1]")

    script_dir = Path(__file__).resolve().parent
    log_path = script_dir / "train.log"
    if not log_path.exists():
        print("Cannot find train.log beside this script: {}".format(log_path))
        return 1

    epoch_losses = parse_train_log(log_path)
    if not epoch_losses:
        print("No epoch/loss pairs were parsed from train.log.")
        print("Please check the log format and update the regular expressions in plot_ema_loss.py if needed.")
        return 1

    epochs, raw_losses = compute_epoch_level_loss(epoch_losses)
    ema_losses = compute_ema(raw_losses, args.alpha)

    csv_path = script_dir / "ema_loss_values.csv"
    png_path = script_dir / "ema_loss_curve.png"
    write_csv(csv_path, epochs, raw_losses, ema_losses)
    plot_curve(png_path, epochs, raw_losses, ema_losses, args.alpha)

    print("Parsed {} epochs from {}".format(len(epochs), log_path))
    print("Wrote {}".format(csv_path))
    print("Wrote {}".format(png_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
