#%%
import torch


#%%
x1 = torch.arange(6).view(2, 3)

#%%
print(x1)

#%%
x1[:, 1] += 1

#%%
print(x1)

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#%%
# create a random 2D tensor and then add a dummy dimension of size 1 at index 0
x = torch.rand(3, 3)
print(x)
x.unsqueeze_(0)
print(x)

#%%
# remove the extra dimension from the previous tensor
print(x)
x.squeeze_(0)
print(x)

#%%
# create a random tensor of shape 5x3 in the interval [3, 7)
x = 3 + torch.rand(5, 3)*(7 - 3)
print(x)

#%%
x = torch.randn(2, 3)
print(x)

#%%
# retrieve the indexes of all the nonzero elements in the tensor torch.Tensor([1, 1, 1, 0, 1])
x = torch.Tensor([1, 1, 1, 0, 1])
print(x)
print(torch.nonzero(x))

#%%
# create a random tensor of shape (2, 1) and then stack 3 copies of that tensor together
x = torch.rand(2, 1)
print(x)
print(torch.stack([x, x, x]))

#%%
# compute the batch matrix-matrix product of 2 matrices
a = torch.rand(3, 2, 3)
b = torch.rand(3, 3, 2)
print(torch.bmm(a, b))

#%%
# compute a batch matrix-matrix product of a 3D matrix and a 2D matrix
a = torch.rand(3, 2, 3)
b = torch.rand(3, 2)
# with broadcasting
print(torch.matmul(a, b))
# without broadcasting
print(torch.bmm(a, b.unsqueeze(0).expand(a.size(0), *b.size())))

#%%
