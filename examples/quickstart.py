"""Offline example using the public implementation; see README for evidence scope."""
import sys,json,pathlib,os
repo=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'src'));os.chdir(repo)
import torch
from model import MyModel
torch.set_num_threads(2); torch.manual_seed(7)
model=MyModel(num_classes=7,dropout=.3)
logits=model(torch.randn(2,3,64,64))
loss=torch.nn.functional.cross_entropy(logits,torch.tensor([1,3])); loss.backward()
print(json.dumps({"fixture":"random RGB tensor, no image accuracy claim","logits_shape":list(logits.shape),"loss":float(loss.detach()),"finite_conv_gradient":bool(torch.isfinite(model.features[0].weight.grad).all())}))
