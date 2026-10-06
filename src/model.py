"""
model.py
========
Builds the segmentation network: a U-Net with a ResNet-34 encoder.

WHY U-NET (analogy)
-------------------
Think of the U-Net as an artist who first SQUINTS at a photo to understand
"there is a blob roughly here" (the encoder / down path), then carefully
PAINTS the exact outline back at full resolution (the decoder / up path).
The "skip connections" are like the artist keeping the original sketch on
the desk so fine edges are not lost while zooming back out.

WHY A PRETRAINED ENCODER
------------------------
ResNet-34 was already trained on 1.2 million everyday photos (ImageNet).
It therefore already knows edges, textures and shapes. We reuse that
"visual common sense" and only teach it the new task (find tumours). This
is called transfer learning and it is the main reason our model converges
fast and scores higher than a network started from random noise.

The network outputs RAW LOGITS (one channel). We deliberately do NOT put a
sigmoid inside the model — applying it later (in the loss with
`from_logits=True`, and manually at inference) is numerically more stable.
"""

import warnings

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


def build_model(
    encoder_name: str = "resnet34",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Create a U-Net segmentation model.

    Parameters
    ----------
    encoder_name : str
        Backbone for the encoder. Default "resnet34".
    encoder_weights : str | None
        "imagenet" for transfer learning, or None to start from scratch
        (used as the baseline in evaluate.py).
    in_channels : int
        Number of input channels. The LGG .tif slices are RGB -> 3.
    classes : int
        Number of output channels. Binary tumour mask -> 1.

    Returns
    -------
    torch.nn.Module
        An untrained U-Net ready for training or loading weights.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=None,  # output raw logits; we apply sigmoid later
    )
    return model


def load_trained_model(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    encoder_name: str = "resnet34",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    """Build the architecture and load saved weights for inference.

    Note: we build with `encoder_weights=None` here because we are about to
    overwrite every weight with the trained checkpoint anyway, so there is no
    need to download ImageNet weights first.
    """
    model = build_model(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=in_channels,
        classes=classes,
    )
    state = torch.load(checkpoint_path, map_location=device)
    # Support both "raw state_dict" and "{'model_state': ...}" checkpoints.
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Standalone self-test: needs NO dataset.
    # We try ImageNet weights first; if the download is blocked (e.g. an
    # offline machine) we fall back to random init so the test still passes.
    # ------------------------------------------------------------------
    warnings.filterwarnings("ignore")
    print("=" * 60)
    print("model.py self-test")
    print("=" * 60)

    try:
        print("Building U-Net (resnet34, imagenet weights)...")
        net = build_model(encoder_weights="imagenet")
        print("  -> ImageNet pretrained weights loaded.")
    except Exception as exc:  # noqa: BLE001
        print(f"  -> Could not download ImageNet weights ({type(exc).__name__}).")
        print("     Falling back to random initialisation for this test.")
        net = build_model(encoder_weights=None)

    print(f"Trainable parameters: {count_parameters(net):,}")

    print("Running a forward pass on a fake 2x3x256x256 batch...")
    dummy = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        out = net(dummy)
    print(f"  -> Output shape: {tuple(out.shape)} (expected (2, 1, 256, 256))")
    assert out.shape == (2, 1, 256, 256), "Unexpected output shape!"

    print("\nSUCCESS: model builds and runs correctly.")
