from einops import repeat
import torch

x = torch.randn(10, 32, 32)
y = repeat(x, 'b h w -> b c h w', c=3)

def desc(inst):
    print(inst)
    print(type(inst))
    print(len(inst))

def test_f(**kwargs):
    desc(kwargs)
    if True:
        pass


#test_f()
tensor = torch.ones(2,3,9)
tensor[:,1] = 0
print(tensor)
n=4
small_pe=tensor[:]
pe=tensor[:,:2]
print(pe)
#输出

print('b')