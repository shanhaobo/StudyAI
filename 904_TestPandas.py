###################################
import os
OutputPath = "output/{}".format(os.path.splitext(os.path.basename(__file__))[0])
os.makedirs(OutputPath, exist_ok=True)
###################################
