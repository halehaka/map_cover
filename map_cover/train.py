from Final_Code.train_FloodNET import train_FloodNET
from Final_Code.train_LandCover import train_LandCover
from Final_Code.train_FloodNET_LandCover import train_FloodNET_LandCover

# TODO: Before Running this code, please run split.py given in landcover.ai.v1 folder

# TODO: set this paths to your datasets paths (to FloodNETresized & landcover.ai.v1)
LandCover_path = "../landcover.ai.v1"
FloodNET_path = "./Israel Pictures/FloodNETresized"

#NOTE: you will get 3 models: trained with LR = 10e-2, retrained with LR = 10e-4, retrained-twice with LR = 10e-5
# TODO: pick a name for your model
model_name = "FloodNET-test"

# TODO: pick a dataset to train on: LandCover, FloodNET or FloodNET_LandCover (union of the datasets)
train_on = "FloodNET"

if train_on == "LandCover":
    train_LandCover(LandCover_path, model_name)

if train_on == "FloodNET":
    train_FloodNET(FloodNET_path, model_name)

if train_on == "FloodNET_LandCover":
    train_FloodNET_LandCover(FloodNET_path, LandCover_path, model_name)

# That's it, now you can see your 3 outputs and go test them in test.py
