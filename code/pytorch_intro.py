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
