import nn
from engine import Tensor

class Model(nn.Module):
    def __init__(self, in_feat, out_feat):
        self.layer1=nn.Linear(in_feat, 128)
        self.layer2=nn.Linear(128, 64)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
model=Model(10, 1)

out=model(Tensor([[1.0]*10]))
print(out)