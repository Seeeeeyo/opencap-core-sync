from matplotlib import pyplot as plt
from utilsChecker import cross_corr_multiple_timeseries
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R
import re


def rotate_dataframe(df, axis, value):
    """
    Rotate the marker data in a DataFrame.

    Parameters:
        df : pd.DataFrame
            DataFrame containing marker position data with `X`, `Y`, and `Z` columns.
        axis : str
            Rotation axis ('x', 'y', 'z').
        value : float
            Angle in degrees.

    Returns:
        pd.DataFrame
            DataFrame with rotated marker data.
    """
    # Get all marker columns by identifying those ending in '-X', '-Y', '-Z'
    # convert the columns names to string
    df.columns = df.columns.astype(str)

    new_column_names = {}
    for col in df.columns:
        if re.search(r'-[XYZ]\d+$', col):
            # Remove trailing numbers from the marker suffix
            new_name = re.sub(r'(\-[XYZ])\d+$', r'\1', col)
            new_column_names[col] = new_name

    # Apply renaming
    df = df.rename(columns=new_column_names)

    # Identify all marker columns ending with '-X', '-Y', or '-Z'
    marker_columns = [col for col in df.columns if re.search(r'-[XYZ]$', col)]

    # Extract unique marker names (e.g., 'sternum', 'r_shoulder', etc.)
    marker_names = list(set(re.sub(r'-[XYZ]$', '', col) for col in marker_columns))

    # Rotate each marker's X, Y, Z components
    for marker in marker_names:
        temp = np.zeros((len(df), 3))
        temp[:, 0] = df[f'{marker}-X']
        temp[:, 1] = df[f'{marker}-Y']
        temp[:, 2] = df[f'{marker}-Z']

        # Create the rotation
        r = R.from_euler(axis, value, degrees=True)
        temp_rot = r.apply(temp)

        # Update the DataFrame with rotated coordinates
        df[f'{marker}-X'] = temp_rot[:, 0]
        df[f'{marker}-Y'] = temp_rot[:, 1]
        df[f'{marker}-Z'] = temp_rot[:, 2]

    return df


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
        engine='python'
    )

    # Replace the column names containing "Unnamed : number" with the previous column name
    for i, col in enumerate(data.columns):
        if 'Unnamed' in col:
            data.columns.values[i] = data.columns.values[i - 1]

    data.columns = data.columns + '-' + data.iloc[0]
    data.columns.values[0] = 'Frame#'
    data.columns.values[1] = 'Time'

    # Drop the first row
    data = data.drop(0)

    return data


def main():
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

                    marker_video_path = os.path.join(marker_video_path, os.listdir(marker_video_path)[0])
                    marker_video_path = os.path.join(marker_video_path,
                                                     [f for f in os.listdir(marker_video_path) if f.endswith('.trc')][0])
                    if not os.path.exists(marker_video_path):
                        continue

                    movement_file_name_trc = movement + '.trc'
                    marker_mocap_path = os.path.join(MarkerDataDir, movement_file_name_trc)

                    try:
                        marker_video_data = read_trc_file(marker_video_path)
                        marker_mocap_data = read_trc_file(marker_mocap_path)
                        marker_video_data = marker_video_data.astype(float)
                        marker_mocap_data = marker_mocap_data.astype(float)
                    except Exception as e:
                        print(f'Error: {e}')
                        continue


                    lag_file_name = f"lag_correlation_{movement}.txt"
                    lag_file_path = os.path.join(movement_path, 'OpenSim', 'IK', 'shiftedIK', lag_file_name)
                    if not os.path.exists(lag_file_path):
                        continue

                    from scipy.interpolate import interp1d

                    # Assuming `marker_video_data` and `marker_mocap_data` are already loaded DataFrames

                    with open(lag_file_path, 'r') as file:
                        lag = int(file.readline().split(':')[-1])
                        correlation = float(file.readline().split(':')[-1])


                    # Convert 'Time' to Timedelta and set as index
                    marker_video_data['time_'] = pd.to_timedelta(marker_video_data['Time'], unit='s')
                    marker_video_data = marker_video_data.set_index('time_')

                    # Upsample the data to 100Hz
                    target_freq = '10ms'  # 100Hz = 10ms intervals
                    marker_video_data = marker_video_data.resample(target_freq).interpolate(method="linear")

                    # Reset the index to make 'time_' accessible as a column again
                    marker_video_data = marker_video_data.reset_index()

                    # Convert 'time_' back to seconds
                    marker_video_data['Time'] = marker_video_data['time_'].dt.total_seconds()

                    # Round the 'Time' column to 2 decimal places
                    marker_video_data['Time'] = marker_video_data['Time'].round(2)

                    # Drop the intermediate 'time_' column
                    marker_video_data = marker_video_data.drop(columns=['time_'])

                    num_rows_mocap = marker_mocap_data.shape[0]
                    num_rows_video = marker_video_data.shape[0]

                    # Adjust and slice marker_video_data based on lag
                    # shifted_video_data is the length of the mocap data and the columns of the video data
                    # shifted_video_data = pd.DataFrame(columns=marker_video_data.columns, index=range(num_rows_mocap))
                    if lag > 0:
                        start = lag
                        end = min(num_rows_video + lag, num_rows_mocap)
                        shifted_video_data = marker_video_data.iloc[:end - lag].reset_index(drop=True)
                    elif lag < 0:
                        start = 0
                        end = min(num_rows_video, num_rows_mocap + lag)
                        shifted_video_data = marker_video_data.iloc[-lag:end - lag].reset_index(drop=True)
                    else:
                        start = 0
                        end = min(num_rows_video, num_rows_mocap)
                        shifted_video_data = marker_video_data.iloc[:end].reset_index(drop=True)

                    # get the time of mocap at the start and end of the video data
                    mocap_time_start = marker_mocap_data['Time'].iloc[start]

                    # Trim to match the length of mocap data
                    shifted_video_data = shifted_video_data.iloc[:num_rows_mocap]

                    # Extract time vectors
                    # time_mocap = marker_mocap_data['Time'].values
                    time_video = shifted_video_data['Time'].values


                    time_offset = mocap_time_start - time_video[0]

                    # add the time offset to the time video
                    time_video += time_offset

                    # round the time to 2 decimal places
                    time_video = np.round(time_video, 2)

                    # set the time of the video data to the time of the mocap data
                    shifted_video_data['Time'] = time_video

                    # trim marker_mocap_data to match the length of shifted_video_data based on the Time column
                    marker_mocap_data_trimmed = marker_mocap_data[marker_mocap_data['Time'].isin(time_video)]

                    # rotational alignment
                    # Calculate midpoints for mono
                    mid_PSIS_mono = np.array([
                        (shifted_video_data['r_PSIS-X15'] + shifted_video_data['l_PSIS-X16']) / 2,
                        (shifted_video_data['r_PSIS-Y15'] + shifted_video_data['l_PSIS-Y16']) / 2,
                        (shifted_video_data['r_PSIS-Z15'] + shifted_video_data['l_PSIS-Z16']) / 2
                    ]).T

                    mid_ASIS_mono = np.array([
                        (shifted_video_data['r_ASIS-X13'] + shifted_video_data['l_ASIS-X14']) / 2,
                        (shifted_video_data['r_ASIS-Y13'] + shifted_video_data['l_ASIS-Y14']) / 2,
                        (shifted_video_data['r_ASIS-Z13'] + shifted_video_data['l_ASIS-Z14']) / 2
                    ]).T

                    # Calculate midpoints for mocap
                    mid_PSIS_mocap = np.array([
                        (marker_mocap_data_trimmed['r.PSIS-X28'] + marker_mocap_data_trimmed['L.PSIS-X30']) / 2,
                        (marker_mocap_data_trimmed['r.PSIS-Y28'] + marker_mocap_data_trimmed['L.PSIS-Y30']) / 2,
                        (marker_mocap_data_trimmed['r.PSIS-Z28'] + marker_mocap_data_trimmed['L.PSIS-Z30']) / 2
                    ]).T

                    mid_ASIS_mocap = np.array([
                        (marker_mocap_data_trimmed['r.ASIS-X27'] + marker_mocap_data_trimmed['L.ASIS-X29']) / 2,
                        (marker_mocap_data_trimmed['r.ASIS-Y27'] + marker_mocap_data_trimmed['L.ASIS-Y29']) / 2,
                        (marker_mocap_data_trimmed['r.ASIS-Z27'] + marker_mocap_data_trimmed['L.ASIS-Z29']) / 2
                    ]).T

                    # Calculate heading vectors
                    heading_vec_mono = mid_ASIS_mono - mid_PSIS_mono
                    heading_vec_mocap = mid_ASIS_mocap - mid_PSIS_mocap

                    # Project into the XZ plane (set Y to 0)
                    heading_vec_mono[:, 1] = 0
                    heading_vec_mocap[:, 1] = 0

                    # Normalize the vectors
                    heading_vec_mono_normalized = heading_vec_mono / np.linalg.norm(heading_vec_mono, axis=1, keepdims=True)
                    heading_vec_mocap_normalized = heading_vec_mocap / np.linalg.norm(heading_vec_mocap, axis=1,
                                                                                      keepdims=True)

                    # Calculate the angular difference using the dot product
                    dot_products = np.einsum('ij,ij->i', heading_vec_mono_normalized, heading_vec_mocap_normalized)
                    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))  # Clip for numerical stability

                    # Convert to degrees if needed
                    angles_degrees = np.degrees(angles)

                    # Calculate the average angular difference
                    average_difference = np.mean(angles_degrees)

                    print(f"Average Angular Difference: {average_difference:.2f} degrees")

                    # Rotate the mono marker data around the Y axis by the average angle in the XZ plane using rotate method
                    rotated_marker_video_data = rotate_dataframe(shifted_video_data, 'y', average_difference)
                    # print(rotated_marker_data)

                    # Translation alignment
                    # Calculate the offset for the midpoints of the ASIS markers at timestep #1 of mocap
                    # Right knee
                    r_knee_mocap = np.array([
                        marker_mocap_data_trimmed['r_knee-X1'],
                        marker_mocap_data_trimmed['r_knee-Y1'],
                        marker_mocap_data_trimmed['r_knee-Z1']
                    ]).T
                    r_knee_mono = np.array([
                        rotated_marker_video_data['r_knee-X17'],
                        rotated_marker_video_data['r_knee-Y17'],
                        rotated_marker_video_data['r_knee-Z17']
                    ]).T

                    # Left knee
                    l_knee_mocap = np.array([
                        marker_mocap_data_trimmed['L_knee-X7'],
                        marker_mocap_data_trimmed['L_knee-Y7'],
                        marker_mocap_data_trimmed['L_knee-Z7']
                    ]).T
                    l_knee_mono = np.array([
                        rotated_marker_video_data['l_knee-X18'],
                        rotated_marker_video_data['l_knee-Y18'],
                        rotated_marker_video_data['l_knee-Z18']
                    ]).T

                    # Right ankle
                    r_ankle_mocap = np.array([
                        marker_mocap_data_trimmed['r_ankle-X21'],
                        marker_mocap_data_trimmed['r_ankle-Y21'],
                        marker_mocap_data_trimmed['r_ankle-Z21']
                    ]).T
                    r_ankle_mono = np.array([
                        rotated_marker_video_data['r_ankle-X19'],
                        rotated_marker_video_data['r_ankle-Y19'],
                        rotated_marker_video_data['r_ankle-Z19']
                    ]).T

                    # Left ankle
                    l_ankle_mocap = np.array([
                        marker_mocap_data_trimmed['L_ankle-X9'],
                        marker_mocap_data_trimmed['L_ankle-Y9'],
                        marker_mocap_data_trimmed['L_ankle-Z9']
                    ]).T
                    l_ankle_mono = np.array([
                        rotated_marker_video_data['l_ankle-X22'],
                        rotated_marker_video_data['l_ankle-Y22'],
                        rotated_marker_video_data['l_ankle-Z22']
                    ]).T

                    # Right shoulder
                    r_shoulder_mocap = np.array([
                        marker_mocap_data_trimmed['R_shoulder-X32'],
                        marker_mocap_data_trimmed['R_shoulder-Y32'],
                        marker_mocap_data_trimmed['R_shoulder-Z32']
                    ]).T

                    r_shoulder_mono = np.array([
                        rotated_marker_video_data['r_shoulder-X2'],
                        rotated_marker_video_data['r_shoulder-Y2'],
                        rotated_marker_video_data['r_shoulder-Z2']
                    ]).T

                    # Left shoulder
                    l_shoulder_mocap = np.array([
                        marker_mocap_data_trimmed['L_shoulder-X33'],
                        marker_mocap_data_trimmed['L_shoulder-Y33'],
                        marker_mocap_data_trimmed['L_shoulder-Z33']
                    ]).T

                    l_shoulder_mono = np.array([
                        rotated_marker_video_data['l_shoulder-X3'],
                        rotated_marker_video_data['l_shoulder-Y3'],
                        rotated_marker_video_data['l_shoulder-Z3']
                    ]).T

                    # Calculate the average offsets
                    # ASIS
                    offset_x_asis = mid_ASIS_mocap[0, 0] - mid_ASIS_mono[0, 0]
                    offset_y_asis = mid_ASIS_mocap[0, 1] - mid_ASIS_mono[0, 1]
                    offset_z_asis = mid_ASIS_mocap[0, 2] - mid_ASIS_mono[0, 2]

                    # PSIS
                    offset_x_psis = mid_PSIS_mocap[0, 0] - mid_PSIS_mono[0, 0]
                    offset_y_psis = mid_PSIS_mocap[0, 1] - mid_PSIS_mono[0, 1]
                    offset_z_psis = mid_PSIS_mocap[0, 2] - mid_PSIS_mono[0, 2]

                    # Right knee
                    offset_x_r_knee = r_knee_mocap[0, 0] - r_knee_mono[0, 0]
                    offset_y_r_knee = r_knee_mocap[0, 1] - r_knee_mono[0, 1]
                    offset_z_r_knee = r_knee_mocap[0, 2] - r_knee_mono[0, 2]

                    # Left knee
                    offset_x_l_knee = l_knee_mocap[0, 0] - l_knee_mono[0, 0]
                    offset_y_l_knee = l_knee_mocap[0, 1] - l_knee_mono[0, 1]
                    offset_z_l_knee = l_knee_mocap[0, 2] - l_knee_mono[0, 2]

                    # Right ankle
                    offset_x_r_ankle = r_ankle_mocap[0, 0] - r_ankle_mono[0, 0]
                    offset_y_r_ankle = r_ankle_mocap[0, 1] - r_ankle_mono[0, 1]
                    offset_z_r_ankle = r_ankle_mocap[0, 2] - r_ankle_mono[0, 2]

                    # Left ankle
                    offset_x_l_ankle = l_ankle_mocap[0, 0] - l_ankle_mono[0, 0]
                    offset_y_l_ankle = l_ankle_mocap[0, 1] - l_ankle_mono[0, 1]
                    offset_z_l_ankle = l_ankle_mocap[0, 2] - l_ankle_mono[0, 2]

                    # Right shoulder
                    offset_x_r_shoulder = r_shoulder_mocap[0, 0] - r_shoulder_mono[0, 0]
                    offset_y_r_shoulder = r_shoulder_mocap[0, 1] - r_shoulder_mono[0, 1]
                    offset_z_r_shoulder = r_shoulder_mocap[0, 2] - r_shoulder_mono[0, 2]

                    # Left shoulder
                    offset_x_l_shoulder = l_shoulder_mocap[0, 0] - l_shoulder_mono[0, 0]
                    offset_y_l_shoulder = l_shoulder_mocap[0, 1] - l_shoulder_mono[0, 1]
                    offset_z_l_shoulder = l_shoulder_mocap[0, 2] - l_shoulder_mono[0, 2]


                    # Calculate the average offsets
                    avg_x_offset = np.mean([
                        offset_x_asis, offset_x_psis, offset_x_r_knee, offset_x_l_knee, offset_x_r_ankle, offset_x_l_ankle,
                        offset_x_r_shoulder, offset_x_l_shoulder
                    ])
                    avg_y_offset = np.mean([
                        offset_y_asis, offset_y_psis, offset_y_r_knee, offset_y_l_knee, offset_y_r_ankle, offset_y_l_ankle,
                        offset_y_r_shoulder, offset_y_l_shoulder
                    ])
                    avg_z_offset = np.mean([
                        offset_z_asis, offset_z_psis, offset_z_r_knee, offset_z_l_knee, offset_z_r_ankle, offset_z_l_ankle,
                        offset_z_r_shoulder, offset_z_l_shoulder
                    ])

                    print(f"Average X Offset: {avg_x_offset:.2f}")
                    print(f"Average Y Offset: {avg_y_offset:.2f}")
                    print(f"Average Z Offset: {avg_z_offset:.2f}")

                    # Apply the offset to the rotated marker data
                    rotated_marker_video_data['r_ASIS-X13'] += avg_x_offset


                    # TODO write market errors to file

                    # TODO write the aligned mono data to file _sync.trc

if __name__ == '__main__':
    main()

