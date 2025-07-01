import torch

def test_cuda_transfer():
    # Create random tensor on cuda:0 with requires_grad=True to mimic your case
    x = torch.randn(5, 5, device='cuda:0', dtype=torch.float32, requires_grad=True)
    print("Original x on cuda:0")
    print(x)
    print(f"dtype: {x.dtype}, min: {x.min()}, max: {x.max()}, mean: {x.mean()}")
    print(f"Is contiguous? {x.is_contiguous()}")
    print(f"Requires grad? {x.requires_grad}")

    # Try moving to cuda:1 directly
    try:
        x1 = x.to('cuda:1')
        print("\nMoved x to cuda:1")
        print(x1)
        print(f"min: {x1.min()}, max: {x1.max()}, mean: {x1.mean()}")
    except Exception as e:
        print(f"Direct .to('cuda:1') failed with error: {e}")

    # Try clone + contiguous + detach + move
    try:
        x2 = x.clone().contiguous().detach().to('cuda:1')
        print("\nClone + contiguous + detach + to('cuda:1')")
        print(x2)
        print(f"min: {x2.min()}, max: {x2.max()}, mean: {x2.mean()}")
    except Exception as e:
        print(f"Clone+contiguous+detach move failed with error: {e}")

if __name__ == "__main__":
    test_cuda_transfer()
