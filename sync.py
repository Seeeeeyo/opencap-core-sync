from utilsChecker import cross_corr_multiple_timeseries
import os
import pandas as pd
import numpy as np


def read_mot_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    for i, line in enumerate(lines):
        if line.startswith('time'):
            start_line = i
            break

    data = pd.read_csv(file_path, delimiter='\t', skiprows=start_line)
    return data


def pad_signals(Y1, Y2):
    print(Y1.shape, Y2.shape)
    max_len = max(Y1.shape[1], Y2.shape[1])
    if Y1.shape[1] < max_len:
        Y1 = np.pad(Y1, ((0, 0), (0, max_len - Y1.shape[1])), 'constant')
    if Y2.shape[1] < max_len:
        Y2 = np.pad(Y2, ((0, 0), (0, max_len - Y2.shape[1])), 'constant')
    print("new shapes:")
    print(Y1.shape, Y2.shape)
    return Y1, Y2


def shift_time_series(Y1, Y2, lag):
    if lag > 0:
        shifted_Y2 = np.roll(Y2, lag, axis=1)
        shifted_Y1 = Y1
    elif lag < 0:
        shifted_Y1 = np.roll(Y1, -lag, axis=1)
        shifted_Y2 = Y2
    else:
        shifted_Y1 = Y1
        shifted_Y2 = Y2
    return shifted_Y1, shifted_Y2


repo_path = '/home/selim/opencap-mono'
validation_videos_path = os.path.join(repo_path, 'LabValidation_withVideos1')
output_path = os.path.join(repo_path, 'output')

subjects_dirs = os.listdir(output_path)

for subject in subjects_dirs:
    OpenSimDataDir = os.path.join(validation_videos_path, subject, 'OpenSimData')
    mocap_dir = os.path.join(OpenSimDataDir, 'Mocap', 'IK')
    subject_path = os.path.join(output_path, subject)
    sessions = os.listdir(subject_path)

    for session in sessions:
        cameras = os.listdir(os.path.join(subject_path, session))

        for camera in cameras:
            movements = os.listdir(os.path.join(subject_path, session, camera))

            for movement in movements:
                movement_path = os.path.join(subject_path, session, camera, movement, movement)
                openSim_video_path = os.path.join(movement_path, 'OpenSim')
                video_ik_path = os.path.join(openSim_video_path, 'IK')

                shifted_dir = os.path.join(video_ik_path, 'shiftedIK')
                if not os.path.exists(shifted_dir):
                    os.makedirs(shifted_dir)

                video_dirs = os.listdir(video_ik_path)

                for video_dir in video_dirs:
                    video_dir_path = os.path.join(video_ik_path, video_dir)
                    video_mot_files = [f for f in os.listdir(video_dir_path) if f.endswith('.mot')]
                    modified_video_mot_files = [f.split('_')[0] + '.mot' for f in video_mot_files]

                    for i, modified_video_file in enumerate(modified_video_mot_files):
                        if "shifted" in modified_video_file:
                            continue

                        video_file = video_mot_files[i]
                        mocap_file_path = os.path.join(mocap_dir, modified_video_file)
                        video_file_path = os.path.join(video_dir_path, video_file)

                        if "shifted" in video_file_path:
                            continue

                        print(f'Mocap file: {mocap_file_path}')
                        print(f'Video file: {video_file_path}')

                        mocap_data = read_mot_file(mocap_file_path)
                        video_data = read_mot_file(video_file_path)

                        mocap_df = pd.DataFrame(mocap_data)
                        video_df = pd.DataFrame(video_data)

                        knee_ankle_columns = ['knee_angle_r', 'knee_angle_l', 'ankle_angle_r', 'ankle_angle_l']
                        mocap_knee_ankle = mocap_df[knee_ankle_columns].values.T
                        video_knee_ankle = video_df[knee_ankle_columns].values.T

                        mocap_knee_ankle, video_knee_ankle = pad_signals(mocap_knee_ankle, video_knee_ankle)

                        max_corr, lag = cross_corr_multiple_timeseries(mocap_knee_ankle, video_knee_ankle,
                                                                       visualize=False)

                        max_corr = round(max_corr, 2)
                        if max_corr > 0.85:
                            print(f'Max Correlation: \033[92m{max_corr}\033[0m')
                        elif max_corr > 0.7:
                            print(f'Max Correlation: \033[93m{max_corr}\033[0m')
                        else:
                            print(f'Max Correlation: \033[91m{max_corr}\033[0m')

                        print(f'Lag: \033[94m{lag}\033[0m')

                        shifted_mocap, shifted_video = shift_time_series(mocap_df.values.T, video_df.values.T, lag)

                        shifted_mocap_df = pd.DataFrame(shifted_mocap.T, columns=mocap_df.columns)
                        shifted_video_df = pd.DataFrame(shifted_video.T, columns=video_df.columns)

                        shifted_mocap_df.to_csv(os.path.join(shifted_dir, f'shifted_mocap_{modified_video_file}'),
                                                sep='\t', index=False)
                        shifted_video_df.to_csv(os.path.join(shifted_dir, f'shifted_mono_{modified_video_file}'),
                                                sep='\t', index=False)

                        with open(os.path.join(shifted_dir, f'lag_correlation_{modified_video_file}.txt'), 'w') as f:
                            f.write(f'Lag: {lag}\nCorrelation: {max_corr}')

                        print('--------------------------------------------')