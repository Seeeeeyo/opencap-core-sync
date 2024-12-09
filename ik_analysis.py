import os
import pandas as pd
from utils_mot import load_mot
import numpy as np

# load four .mot files
root_path = "/home/selim/opencap-mono"
# mot_paths = {
#     "mocap": "Reference/Mocap/IK/squats1_compareToMonocular.mot",
#     "video_2cam": "Reference/Video_2cam/IK/squats1_compareToMonocular.mot",
#     "wham_opt": "IK/squats1_5/squats1_5_compareToMonocular.mot",
#     "wham_no_opt": "IK/squats1_5_wham_result/squats1_5_wham_result_compareToMonocular.mot",
# }

mot_paths = {
    "mocap": "LabValidation_withVideos1/subject3/OpenSimData/Mocap/IK/STS1.mot",
    "mono": "output/subject3/Session0/Cam1/STS1/STS1/OpenSim/IK/shiftedIK/STS1_5_sync.mot",
    # "wham_opt": "/home/selim/opencap-mono/LabValidation_withVideos1/subject3/OpenSimData/IK",
}

output_path = os.path.join(root_path, 'output/subject3/Session0/Cam1/STS1/STS1/OpenSim/IK/shiftedIK/')

mot_data = {
    key: load_mot(os.path.join(root_path, mot_paths[key]))
    for key in mot_paths.keys()
}

mocap = mot_data["mocap"]
mono = mot_data["mono"]

# convert to df
mocap = pd.DataFrame(data=mocap[0], columns=mocap[1])
mono = pd.DataFrame(data=mono[0], columns=mono[1])

# trim mocap to match mono length
mocap = mocap[mocap["time"] <= mono["time"].iloc[-1]]

assert mocap.shape[0] == mono.shape[0], "Lengths do not match"

# find the common columns between the two dataframes
common_columns = list(set(mocap.columns).intersection(mono.columns))

common_columns.remove("time")

# for each common_columns, calculate the mean squared error
mse_global = 0
for col in common_columns:
    mse = np.round(((mocap[col] - mono[col]) ** 2).mean(),1)
    mse_global += mse
    print(f"{col}: {mse} degrees.")

mse_global /= len(common_columns)
mse_global = np.round(mse_global, 1)

print(f"Global MSE: {mse_global} degrees.")

# write the results to a file
with open(os.path.join(output_path, 'translation_error.txt'), 'w') as f:
    f.write(f"Global MSE: {mse_global} degrees.\n")
    for col in common_columns:
        mse = np.round(((mocap[col] - mono[col]) ** 2).mean(),1)
        f.write(f"{col}: {mse} degrees.\n")
