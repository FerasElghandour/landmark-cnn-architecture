"""Train or evaluate the released model using fixed prepared data splits."""
from model import MyModel
from training import main

def build_model(c):
    return MyModel(num_classes=c["classes"],dropout=c["dropout"])

if __name__=="__main__":
    main(build_model,{'classes': 50, 'dropout': 0.7},binary=False,tokens=False)
