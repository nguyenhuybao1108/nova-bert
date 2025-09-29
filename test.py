import torch

# Element-wise multiplication (Hadamard product)
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
# require same shape
print("Element-wise multiplication (a * b):") 
print(a * b)
# Output: [[5, 12],
#          [21, 32]]

# Matrix multiplication (matmul)
print("\nMatrix multiplication (torch.matmul(a, b)):")
print(torch.matmul(a, b))
# Output:
# [[19, 22],
#  [43, 50]]

# Dot product (1D tensors only)
v1 = torch.tensor([1.0, 2.0, 3.0])
v2 = torch.tensor([4.0, 5.0, 6.0])

print("\nDot product (torch.dot(v1, v2)):")
print(torch.dot(v1, v2))  # Output: 1*4 + 2*5 + 3*6 = 32.0


# Pairwise product = Element-wise multiplication
