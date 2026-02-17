#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


class DummyVideoModel(torch.nn.Module):
    """Fallback stub if you don't have baseline model wiring yet."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = torch.nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export word model to ONNX")
    parser.add_argument("--checkpoint", default="", help="Path to torch checkpoint (optional in this stub)")
    parser.add_argument("--labels", required=True, help="labels.txt path")
    parser.add_argument("--out", required=True, help="Output ONNX path")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--window-frames", type=int, default=32)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = [line.strip() for line in Path(args.labels).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not labels:
        raise ValueError("labels file is empty")

    model = DummyVideoModel(num_classes=len(labels))
    model.eval()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)

    x = torch.randn(1, 3, int(args.window_frames), int(args.input_size), int(args.input_size), dtype=torch.float32)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        x,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "time"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    print("[export_onnx] done")
    print(f"  out={out_path}")
    print(f"  classes={len(labels)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
