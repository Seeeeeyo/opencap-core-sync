from matplotlib import pyplot as plt

from utilsChecker import cross_corr_multiple_timeseries
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def read_trc_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Find the line where the data header starts
    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith("Frame#"):
            start_line = i
            break

    # Read the data from the identified starting line
    data = pd.read_csv(
        file_path,
        delimiter='\t',
        skiprows=start_line,
        header=[0, 1],  # Specify double header
        engine='python'
    )
    return data


repo_path = '/home/selim/opencap-mono'
validation_videos_path = os.path.join(repo_path, 'LabValidation_withVideos1')
output_path = os.path.join(repo_path, 'output')

subjects_dirs = os.listdir(output_path)

for subject in subjects_dirs:
    MarkerDataDir = os.path.join(validation_videos_path, subject, 'MarkerData', 'Mocap')
    subject_path = os.path.join(output_path, subject)
    sessions = os.listdir(subject_path)

    for session in sessions:
        cameras = os.listdir(os.path.join(subject_path, session))

        for camera in cameras:
            movements = os.listdir(os.path.join(subject_path, session, camera))

            for movement in movements:
                movement_path = os.path.join(subject_path, session, camera, movement, movement)
                marker_video_path = os.path.join(movement_path, 'MarkerData')
                if not os.path.exists(marker_video_path):
                    continue
                # get the folder in marker_video_path
                marker_video_path = os.path.join(marker_video_path, os.listdir(marker_video_path)[0])
                # get the file path in marker_video_path with the .trc extension
                marker_video_path = os.path.join(marker_video_path, [f for f in os.listdir(marker_video_path) if f.endswith('.trc')][0])
                if not os.path.exists(marker_video_path):
                    continue

                movement_file_name_trc = movement + '.trc'
                marker_mocap_path = os.path.join(MarkerDataDir, movement_file_name_trc)

                try:
                    marker_video_data = read_trc_file(marker_video_path)
                    marker_mocap_data = read_trc_file(marker_mocap_path)
                except Exception as e:
                    print(f'Error: {e}')
                    continue
                lag_file_name = f"lag_correlation_{movement}.txt" # lag_correlation_{video_file}.txt
                lag_file_path = os.path.join(movement_path, 'OpenSim', 'IK', 'shiftedIK', lag_file_name)
                if not os.path.exists(lag_file_path):
                    continue
                with open(lag_file_path, 'r') as file:
                    # the lag is as the following: Lag : 0.000000. the correlation is the next line and as the following: Correlation : 0.999999
                    lag = float(file.readline().split(':')[-1])
                    correlation = float(file.readline().split(':')[-1])

                # TODO: Shift using lag from above, trim to mocap length, interpolate so the mono time vector is the same as the mocap time vector.

                # TODO rotational aligment

                # TODO translation aligment

                # TODO write market errors to file

                # TODO write the aligned mono data to file _sync.trc
