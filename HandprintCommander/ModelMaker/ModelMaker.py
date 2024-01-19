from HandprintCommander.ModelMaker.GestureModelTrainer import train
from HandprintCommander.ModelMaker.ModelConverter import convert

def make():
    train()
    convert()
    
if __name__ == "__main__":
    make()