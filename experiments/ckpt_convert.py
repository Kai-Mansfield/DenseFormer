import torch
import os

def convert_checkpoint(new_ckpt_path, old_ckpt_path=None):
    """
    Converts a checkpoint saved with keys ('model', 'opt', 'scheduler', 'itr')
    into the older format with keys ('model', 'optimizer', 'scheduler', 'itr').
    Creates a new checkpoint file instead of overwriting the original.
    """

    # Load the new-format checkpoint
    ckpt = torch.load(new_ckpt_path, map_location='cpu')

    # Build the converted checkpoint in the old format
    converted = {
        'model': ckpt['model'],
        'optimizer': ckpt.get('opt', {}),
        'scheduler': ckpt.get('scheduler', {}),
        'itr': ckpt.get('itr', 0)
    }

    # Determine save path
    if old_ckpt_path is None:
        base, ext = os.path.splitext(new_ckpt_path)
        old_ckpt_path = f"{base}_converted{ext}"

        # If the converted file already exists, create a unique name
        i = 1
        while os.path.exists(old_ckpt_path):
            old_ckpt_path = f"{base}_converted_{i}{ext}"
            i += 1

    # Save new checkpoint
    torch.save(converted, old_ckpt_path)
    print(f"✅ Converted checkpoint saved to: {old_ckpt_path}")

    return old_ckpt_path
