import torch
from minGRU_pytorch.minGRULM import minGRULM

# Model parameters (same as your training script)
num_tokens = 256
dim = 512 * 7
depth = 8
ff_mult = 4
min_gru_expansion = 1.5
conv_kernel_size = 3

# Create the model
model = minGRULM(
    num_tokens=num_tokens,
    dim=dim,
    depth=depth,
    ff_mult=ff_mult,
    min_gru_expansion=min_gru_expansion,
    conv_kernel_size=conv_kernel_size
)

# Calculate total parameters
total_params = sum(p.numel() for p in model.parameters())

print(f"Total number of parameters: {total_params}")

# Calculate approximate size in MB (assuming float32)
size_mb = total_params * 4 / (1024**2)  # 4 bytes per float32 parameter
print(f"Approximate size (float32): {size_mb:.2f} MB")

# If you save the model with half-precision (float16), the size would be roughly halved.
size_mb_fp16 = size_mb / 2
print(f"Approximate size (float16): {size_mb_fp16:.2f} MB")


# You can also break down the parameter count per layer/module for more detailed analysis:

def count_parameters(module):
    return sum(p.numel() for p in module.parameters())

print("\nParameter breakdown:")
print(f"Embedding: {count_parameters(model.token_emb)}")
for i, layer in enumerate(model.layers):
    print(f"Layer {i+1}:")
    print(f"  Conv: {count_parameters(layer[0])}")
    print(f"  RMSNorm 1: {count_parameters(layer[1])}")
    print(f"  minGRU: {count_parameters(layer[2])}")
    print(f"  RMSNorm 2: {count_parameters(layer[3])}")
    print(f"  FeedForward: {count_parameters(layer[4])}")

print(f"Final RMSNorm: {count_parameters(model.norm)}")
print(f"Logits Layer: {count_parameters(model.to_logits)}")