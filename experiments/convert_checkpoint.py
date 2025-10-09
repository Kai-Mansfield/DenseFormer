import torch
import sys
import os

def convert_checkpoint(old_ckpt_path, new_ckpt_path):
    # --- Sanity check ---
    if not os.path.exists(old_ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {old_ckpt_path}")

    print(f"🔹 Loading checkpoint: {old_ckpt_path}")
    ckpt = torch.load(old_ckpt_path, map_location="cuda:0")

    # --- Extract model, optimizer, scheduler, itr ---
    model_state = ckpt.get('model', ckpt)
    opt_state = ckpt.get('opt', None)
    sched_state = ckpt.get('scheduler', None)
    itr = ckpt.get('itr', 0)

    # --- Clean state dict (remove _orig_mod. prefix) ---
    clean_state = {}
    for k, v in model_state.items():
        new_k = k
        if k.startswith("_orig_mod."):
            new_k = k[len("_orig_mod."):]
        clean_state[new_k] = v

    # --- Construct old-format checkpoint ---
    new_ckpt = {
        'model': clean_state,
        'optimizer': opt_state,
        'scheduler': sched_state,
        'itr': itr,
    }

    # --- Overwrite existing file ---
    if os.path.exists(new_ckpt_path):
        print(f"⚠️  File already exists at {new_ckpt_path}, overwriting...")

    torch.save(new_ckpt, new_ckpt_path)
    print(f"✅ Converted checkpoint saved as: {new_ckpt_path}")

    return new_ckpt_path


# --- Command-line entrypoint ---
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_checkpoint.py <old_ckpt_path> <new_ckpt_path>")
        sys.exit(1)

    old_ckpt_path = sys.argv[1]
    new_ckpt_path = sys.argv[2]
    convert_checkpoint(old_ckpt_path, new_ckpt_path)
