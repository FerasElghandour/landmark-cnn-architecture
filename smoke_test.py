import torch
from model import MyModel
torch.set_num_threads(2)
model = MyModel(num_classes=7, dropout=0.3)
x = torch.randn(2,3,64,64)
logits = model(x)
assert logits.shape == (2,7) and torch.isfinite(logits).all()
torch.nn.functional.cross_entropy(logits, torch.tensor([1,3])).backward()
assert model.features[0].weight.grad is not None
assert torch.isfinite(model.features[0].weight.grad).all()
print('Landmark CNN: output shape and finite end-to-end gradients passed.')
