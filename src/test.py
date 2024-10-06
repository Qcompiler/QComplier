
import torch
a = torch.load("act.pt","cpu").to(torch.float32)
b = torch.load("weight_cache.pt","cpu").to(torch.float32)
print(a)
print(b)
d = torch.mm(a,b.T)
c = torch.max(d)

indices = (d == c).nonzero(as_tuple=True)

# 将结果转化为行和列的索引
rows = indices[0]
cols = indices[1]
print(rows)
print(cols)

print(d[:,4722])
print(a.shape)
print(b.shape)
x = torch.mm(a[8:9,:],b.T[:,4722:4723]).T
print(x)
print(a[8:9,:])
