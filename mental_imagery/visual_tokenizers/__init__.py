#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Frozen image tokenizers: LlamaGen VQ-16 and the FSQ board autoencoder."""

# Pinned LlamaGen weights (Hugging Face Hub). Kept here, free of heavy imports, so tools that
# only need the revision do not import torch.
LLAMAGEN_REPO_ID = "FoundationVision/LlamaGen"
LLAMAGEN_REVISION = "81e41139272c038412e4fe8f1c52a51ebbf95b8b"
LLAMAGEN_VQ16_FILENAME = "vq_ds16_c2i.pt"  # the only file of the repository the code loads
