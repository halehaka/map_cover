from Final_Code.testLandCover import testLandCover
from Final_Code.testFloodNET import testFloodNET

# TODO: Before Running this code, please run split.py given in landcover.ai.v1 folder

# TODO: set this paths to your datasets paths (to FloodNETtestresized & landcover.ai.v1)
LandCover_path = "../landcover.ai.v1"
FloodNET_path = "./Israel Pictures/FloodNETtestresized"

#NOTE: you will get 3 models: trained with LR = 10e-2, retrained with LR = 10e-4, retrained-twice with LR = 10e-5
# TODO: pick a name of a model to test (without .h5)
model_name = "v7-LandCover-retrained-twice"

# TODO: pick a dataset to test on: LandCover or FloodNET
test_on = "LandCover"

# TODO: decide if to show samples and if to save samples to /examples
show_samples = True
save_samples = False

if test_on == "LandCover":
    testLandCover(LandCover_path, model_name, show_samples, save_samples)

if test_on == "FloodNET":
    testFloodNET(FloodNET_path, model_name, show_samples, save_samples)
