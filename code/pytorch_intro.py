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
