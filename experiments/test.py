import torch

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU 0 name:", torch.cuda.get_device_name(0))
    print("GPU 1 name:", torch.cuda.get_device_name(1))

    # Create tensor on cuda:0
    x = torch.randn(5, 5, device='cuda:0', requires_grad=True)
    print("\nOriginal tensor x on cuda:0:")
    print(x)
    print(f"min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}")
    
    # Move tensor x to cuda:1
    torch.cuda.synchronize(0)
    x1 = x.to('cuda:1')
    torch.cuda.synchronize(1)
    
    print("\nTensor x moved to cuda:1:")
    print(x1)
    print(f"min: {x1.min().item()}, max: {x1.max().item()}, mean: {x1.mean().item()}")

    # Try moving CPU->cuda:1 directly for comparison
    x_cpu = torch.randn(5, 5)
    x_gpu1 = x_cpu.to('cuda:1')
    print("\nTensor created on CPU and moved to cuda:1:")
    print(x_gpu1)
    print(f"min: {x_gpu1.min().item()}, max: {x_gpu1.max().item()}, mean: {x_gpu1.mean().item()}")

if __name__ == "__main__":
    main()
