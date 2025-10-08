import torch
import os

def convert_checkpoint(old_ckpt_path, new_ckpt_path=None):
    """
    Converts an 'old' checkpoint saved with keys ('model', 'opt', 'scheduler', 'itr')
    into the 'new' format with keys ('model', 'optimizer', 'scheduler', 'itr').

    The original file is treated as the old-format checkpoint,
    and the converted one is saved as the new-format checkpoint.
    """

    # Load the old-format checkpoint (actually the newer save method)
    ckpt = torch.load(old_ckpt_path, map_location='cpu')

    # Convert to the desired format (used by save_checkpoint)
    new_ckpt = {
        'model': ckpt['model'],
        'optimizer': ckpt.get('opt', {}),
        'scheduler': ckpt.get('scheduler', {}),
        'itr': ckpt.get('itr', 0)
    }

    # Determine output file path
    if new_ckpt_path is None:
        base, ext = os.path.splitext(old_ckpt_path)
        new_ckpt_path = f"{base}_new{ext}"

        # Avoid overwriting
        i = 1
        while os.path.exists(new_ckpt_path):
            new_ckpt_path = f"{base}_new_{i}{ext}"
            i += 1

    # Save the converted checkpoint
    torch.save(new_ckpt, new_ckpt_path)
    print(f"✅ Converted checkpoint saved as: {new_ckpt_path}")

    return new_ckpt_path
