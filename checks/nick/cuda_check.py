import torch

def check_cuda():
    """Check if CUDA is available and print device information."""
    cuda_available = torch.cuda.is_available()

    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("CUDA is not available. Using CPU.")

    return cuda_available

if __name__ == "__main__":
    check_cuda()
