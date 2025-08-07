#!/usr/bin/env python3
"""
PyTorch 2.6 Compatibility Patches

This module provides patches for PyTorch 2.6 compatibility issues,
particularly for TTS libraries that haven't been updated yet.
"""

import logging
import torch

logger = logging.getLogger(__name__)

def apply_pytorch_patches():
    """Apply all PyTorch 2.6 compatibility patches"""
    _patch_torch_load()
    logger.info("Applied PyTorch 2.6 compatibility patches")

def _patch_torch_load():
    """Patch torch.load to handle PyTorch 2.6 weights_only compatibility issue"""
    original_torch_load = torch.load
    
    def patched_torch_load(f, *args, **kwargs):
        # Force weights_only=False for TTS model loading to avoid compatibility issues
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_torch_load(f, *args, **kwargs)
    
    torch.load = patched_torch_load
    logger.info("Applied PyTorch 2.6 weights_only compatibility patch")

# Apply patches when module is imported
apply_pytorch_patches() 