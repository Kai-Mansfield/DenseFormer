import torch

# Paths to your two model checkpoints
path_uninterrupted = "/mnt/lustre/users/inf/kajm20/DenseFormer/experiments/exps/owt2/denseformer2/uninterrupted.pt"
path_resumed = "/mnt/lustre/users/inf/kajm20/DenseFormer/experiments/exps/owt2/denseformer2/resumed.pt"

# Load both (assume they were saved with torch.save(model.state_dict(), ...))
state1 = torch.load(path_uninterrupted, map_location="cpu")
state2 = torch.load(path_resumed, map_location="cpu")

# If you saved full checkpoints, extract the model weights
if "model" in state1:
    state1 = state1["model"]
if "model" in state2:
    state2 = state2["model"]

# Function to compare
def compare_state_dicts(sd1, sd2):
    keys1, keys2 = set(sd1.keys()), set(sd2.keys())
    if keys1 != keys2:
        print("⚠️ Key mismatch between checkpoints!")
        print("Only in model1:", keys1 - keys2)
        print("Only in model2:", keys2 - keys1)
        return False

    all_match = True
    for k in sorted(sd1.keys()):
        v1, v2 = sd1[k], sd2[k]
        if not torch.allclose(v1, v2, atol=0, rtol=0):
            diff = (v1 - v2).abs().max().item()
            print(f"❌ Mismatch in {k}: max abs diff = {diff}")
            all_match = False
    return all_match

# Run the check
identical = compare_state_dicts(state1, state2)

if identical:
    print("✅ Models are bitwise identical!")
else:
    print("⚠️ Models differ (training diverged after resume).")
