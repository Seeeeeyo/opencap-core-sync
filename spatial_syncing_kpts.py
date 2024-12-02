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
from utils_trc import write_trc


def clean_column_names(df):
    """
    Clean the column names of a DataFrame by removing trailing numbers from marker suffixes,
    converting all column names to lowercase except for the '-X', '-Y', and '-Z' suffixes,
    and keeping 'Time' and 'Frame' column names uppercase.

    Parameters:
        df : pd.DataFrame
            DataFrame containing marker position data with `X`, `Y`, and `Z` columns.

    Returns:
        pd.DataFrame
            DataFrame with cleaned column names.
    """
    df.columns = df.columns.astype(str)
    new_column_names = {}
    for col in df.columns:
        if col in ["Time", "Frame#"]:
            new_column_names[col] = col
        elif re.search(r"-[XYZ]\d+$", col):
            new_name = re.sub(r"(\-[XYZ])\d+$", r"\1", col)
            new_name = new_name.lower()[:-2] + new_name[-2:].upper()
            new_column_names[col] = new_name
        else:
            new_column_names[col] = col.lower()
    df = df.rename(columns=new_column_names)
    return df


def extract_alpha_chars(s):
    return "".join(sorted(re.findall(r"[a-zA-Z]", s)))


def compute_marker_errors(rotated_marker_video_data, marker_mocap_data_trimmed):
    """
    Compute the mean per marker error for each marker and the average error over all markers.

    Parameters:
        rotated_marker_video_data : pd.DataFrame
            DataFrame containing the rotated marker video data.
        marker_mocap_data_trimmed : pd.DataFrame
            DataFrame containing the trimmed marker mocap data.

    Returns:
        dict
            Dictionary containing the mean error for each marker.
        float
            Average error over all markers.
    """
    marker_errors = {}
    total_error = 0
    marker_count = 0

    # Extract marker names and their alphabetic characters
    video_markers = {
        col: extract_alpha_chars(col)
        for col in rotated_marker_video_data.columns
        if col not in ["Time", "Frame#"]
    }
    mocap_markers = {
        col: extract_alpha_chars(col) for col in marker_mocap_data_trimmed.columns
    }

    # Find common markers based on the sorted alphabetic characters
    common_markers = set()
    for video_marker, video_alpha in video_markers.items():
        for mocap_marker, mocap_alpha in mocap_markers.items():
            if video_alpha == mocap_alpha:
                common_markers.add((video_marker, mocap_marker))
                break
    print(f"Common Markers: {common_markers}")

    # Find unmatched markers
    matched_video_markers = {pair[0] for pair in common_markers}
    matched_mocap_markers = {pair[1] for pair in common_markers}

    unmatched_video_markers = set(video_markers.keys()) - matched_video_markers
    unmatched_mocap_markers = set(mocap_markers.keys()) - matched_mocap_markers

    # Remove 'Time' and 'Frame#' from unmatched markers
    unmatched_video_markers.discard("Time")
    unmatched_video_markers.discard("Frame#")
    unmatched_mocap_markers.discard("Time")
    unmatched_mocap_markers.discard("Frame#")

    print(f"Unmatched Video Markers: {unmatched_video_markers}")
    print(f"Unmatched Mocap Markers: {unmatched_mocap_markers}")

    # base_common_pairs is a set of common markers from common_markers but without the suffixes -X, -Y, -Z
    base_common_pairs = set()
    for pair in common_markers:
        pair_0 = pair[0][:-2]
        pair_1 = pair[1][:-2]
        if len(pair_0) > 1:
            base_common_pairs.add((pair_0, pair_1))

    for marker in base_common_pairs:
        video_marker_base = marker[0]
        mocap_marker_base = marker[1]

        # Compute the Euclidean distance for each marker
        error = np.sqrt(
            (
                rotated_marker_video_data[f"{video_marker_base}-X"]
                - marker_mocap_data_trimmed[f"{mocap_marker_base}-X"]
            )
            ** 2
            + (
                rotated_marker_video_data[f"{video_marker_base}-Y"]
                - marker_mocap_data_trimmed[f"{mocap_marker_base}-Y"]
            )
            ** 2
            + (
                rotated_marker_video_data[f"{video_marker_base}-Z"]
                - marker_mocap_data_trimmed[f"{mocap_marker_base}-Z"]
            )
            ** 2
        )

        mean_error = np.mean(error)
        marker_errors[video_marker_base] = round(mean_error, 3)
        total_error += mean_error
        marker_count += 1

    # Compute the average error over all markers
    average_error = total_error / marker_count if marker_count > 0 else 0

    return marker_errors, round(average_error, 3)


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
    marker_columns = [col for col in df.columns if re.search(r"-[XYZ]$", col)]
    marker_names = list(set(re.sub(r"-[XYZ]$", "", col) for col in marker_columns))

    for marker in marker_names:
        temp = np.zeros((len(df), 3))
        temp[:, 0] = df[f"{marker}-X"]
        temp[:, 1] = df[f"{marker}-Y"]
        temp[:, 2] = df[f"{marker}-Z"]

        r = R.from_euler(axis, value, degrees=True)
        temp_rot = r.apply(temp)

        df[f"{marker}-X"] = temp_rot[:, 0]
        df[f"{marker}-Y"] = temp_rot[:, 1]
        df[f"{marker}-Z"] = temp_rot[:, 2]

    return df


def read_trc_file(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Find the line where the data header starts
    start_line = 0
    for i, line in enumerate(lines):
        if line.startswith("Frame#"):
            start_line = i
            break

    # Read the data from the identified starting line
    data = pd.read_csv(file_path, delimiter="\t", skiprows=start_line, engine="python")

    # Replace the column names containing "Unnamed : number" with the previous column name
    for i, col in enumerate(data.columns):
        if "Unnamed" in col:
            data.columns.values[i] = data.columns.values[i - 1]

    data.columns = data.columns + "-" + data.iloc[0]
    data.columns.values[0] = "Frame#"
    data.columns.values[1] = "Time"

    # Drop the first row
    data = data.drop(0)

    # clean the column names
    cleaned_data = clean_column_names(data)

    return cleaned_data


def main():
    repo_path = "/home/selim/opencap-mono"
    validation_videos_path = os.path.join(repo_path, "LabValidation_withVideos1")
    output_path = os.path.join(repo_path, "output")

    subjects_dirs = os.listdir(output_path)

    for subject in subjects_dirs:
        MarkerDataDir = os.path.join(
            validation_videos_path, subject, "MarkerData", "Mocap"
        )
        subject_path = os.path.join(output_path, subject)
        sessions = os.listdir(subject_path)

        for session in sessions:
            cameras = os.listdir(os.path.join(subject_path, session))

            for camera in cameras:
                movements = os.listdir(os.path.join(subject_path, session, camera))

                for movement in movements:
                    movement_path = os.path.join(
                        subject_path, session, camera, movement, movement
                    )
                    marker_video_path = os.path.join(movement_path, "MarkerData")
                    error_markers_path = os.path.join(movement_path, "MarkerData")
                    if not os.path.exists(marker_video_path):
                        continue

                    marker_video_subdirs = [
                        d for d in os.listdir(marker_video_path) if d != 'errors'
                    ]
                    if not marker_video_subdirs:
                        continue

                    marker_video_path = os.path.join(marker_video_path, marker_video_subdirs[0])
                    marker_video_files = [
                        f for f in os.listdir(marker_video_path) if f.endswith(".trc")
                    ]
                    if not marker_video_files:
                        continue

                    marker_video_path = os.path.join(marker_video_path, marker_video_files[0])
                    if not os.path.exists(marker_video_path):
                        continue

                    movement_file_name_trc = movement + ".trc"
                    marker_mocap_path = os.path.join(
                        MarkerDataDir, movement_file_name_trc
                    )

                    try:
                        marker_video_data = read_trc_file(marker_video_path)
                        marker_mocap_data = read_trc_file(marker_mocap_path)
                        marker_video_data = marker_video_data.astype(float)
                        marker_mocap_data = marker_mocap_data.astype(float)
                    except Exception as e:
                        print(f"Error: {e}")
                        continue

                    lag_file_name = f"lag_correlation_{movement}.txt"
                    lag_file_path = os.path.join(
                        movement_path, "OpenSim", "IK", "shiftedIK", lag_file_name
                    )
                    if not os.path.exists(lag_file_path):
                        continue

                    from scipy.interpolate import interp1d

                    # Assuming `marker_video_data` and `marker_mocap_data` are already loaded DataFrames
                    with open(lag_file_path, "r") as file:
                        lag = int(file.readline().split(":")[-1])
                        correlation = float(file.readline().split(":")[-1])

                    # Convert 'Time' to Timedelta and set as index
                    marker_video_data["time_"] = pd.to_timedelta(
                        marker_video_data["Time"], unit="s"
                    )
                    marker_video_data = marker_video_data.set_index("time_")

                    # Upsample the data to 100Hz
                    target_freq = "10ms"  # 100Hz = 10ms intervals
                    marker_video_data = marker_video_data.resample(
                        target_freq
                    ).interpolate(method="linear")

                    # Reset the index to make 'time_' accessible as a column again
                    marker_video_data = marker_video_data.reset_index()

                    # Convert 'time_' back to seconds
                    marker_video_data["Time"] = marker_video_data[
                        "time_"
                    ].dt.total_seconds()

                    # Round the 'Time' column to 2 decimal places
                    marker_video_data["Time"] = marker_video_data["Time"].round(2)

                    # Drop the intermediate 'time_' column
                    marker_video_data = marker_video_data.drop(columns=["time_"])

                    num_rows_mocap = marker_mocap_data.shape[0]
                    num_rows_video = marker_video_data.shape[0]

                    # Adjust and slice marker_video_data based on lag
                    # shifted_video_data is the length of the mocap data and the columns of the video data
                    # shifted_video_data = pd.DataFrame(columns=marker_video_data.columns, index=range(num_rows_mocap))
                    if lag > 0:
                        start = lag
                        end = min(num_rows_video + lag, num_rows_mocap)
                        shifted_video_data = marker_video_data.iloc[
                            : end - lag
                        ].reset_index(drop=True)
                    elif lag < 0:
                        start = 0
                        end = min(num_rows_video, num_rows_mocap + lag)
                        shifted_video_data = marker_video_data.iloc[
                            -lag : end - lag
                        ].reset_index(drop=True)
                    else:
                        start = 0
                        end = min(num_rows_video, num_rows_mocap)
                        shifted_video_data = marker_video_data.iloc[:end].reset_index(
                            drop=True
                        )

                    # get the time of mocap at the start and end of the video data
                    mocap_time_start = marker_mocap_data["Time"].iloc[start]

                    # Trim to match the length of mocap data
                    shifted_video_data = shifted_video_data.iloc[:num_rows_mocap]

                    # Extract time vectors
                    # time_mocap = marker_mocap_data['Time'].values
                    time_video = shifted_video_data["Time"].values

                    time_offset = mocap_time_start - time_video[0]

                    # add the time offset to the time video
                    time_video += time_offset

                    # round the time to 2 decimal places
                    time_video = np.round(time_video, 2)

                    # set the time of the video data to the time of the mocap data
                    shifted_video_data["Time"] = time_video

                    # trim marker_mocap_data to match the length of shifted_video_data based on the Time column
                    marker_mocap_data_trimmed = marker_mocap_data[
                        marker_mocap_data["Time"].isin(time_video)
                    ]

                    # rotational alignment
                    # Calculate midpoints for mono
                    right_psis = "r_psis"
                    left_psis = "l_psis"

                    mid_PSIS_mono = np.array(
                        [
                            (
                                shifted_video_data[f"{right_psis}-X"]
                                + shifted_video_data[f"{left_psis}-X"]
                            )
                            / 2,
                            (
                                shifted_video_data[f"{right_psis}-Y"]
                                + shifted_video_data[f"{left_psis}-Y"]
                            )
                            / 2,
                            (
                                shifted_video_data[f"{right_psis}-Z"]
                                + shifted_video_data[f"{left_psis}-Z"]
                            )
                            / 2,
                        ]
                    ).T

                    mid_asis_right = "r_asis"
                    mid_asis_left = "l_asis"
                    mid_ASIS_mono = np.array(
                        [
                            (
                                shifted_video_data[f"{mid_asis_right}-X"]
                                + shifted_video_data[f"{mid_asis_left}-X"]
                            )
                            / 2,
                            (
                                shifted_video_data[f"{mid_asis_right}-Y"]
                                + shifted_video_data[f"{mid_asis_left}-Y"]
                            )
                            / 2,
                            (
                                shifted_video_data[f"{mid_asis_right}-Z"]
                                + shifted_video_data[f"{mid_asis_left}-Z"]
                            )
                            / 2,
                        ]
                    ).T

                    # Calculate midpoints for mocap
                    right_psis_mocap = "r.psis"
                    left_psis_mocap = "l.psis"
                    mid_PSIS_mocap = np.array(
                        [
                            (
                                marker_mocap_data_trimmed[f"{right_psis_mocap}-X"]
                                + marker_mocap_data_trimmed[f"{left_psis_mocap}-X"]
                            )
                            / 2,
                            (
                                marker_mocap_data_trimmed[f"{right_psis_mocap}-Y"]
                                + marker_mocap_data_trimmed[f"{left_psis_mocap}-Y"]
                            )
                            / 2,
                            (
                                marker_mocap_data_trimmed[f"{right_psis_mocap}-Z"]
                                + marker_mocap_data_trimmed[f"{left_psis_mocap}-Z"]
                            )
                            / 2,
                        ]
                    ).T

                    right_mid_asis_mocap = "r.asis"
                    left_mid_asis_mocap = "l.asis"
                    mid_ASIS_mocap = np.array(
                        [
                            (
                                marker_mocap_data_trimmed[f"{right_mid_asis_mocap}-X"]
                                + marker_mocap_data_trimmed[f"{left_mid_asis_mocap}-X"]
                            )
                            / 2,
                            (
                                marker_mocap_data_trimmed[f"{right_mid_asis_mocap}-Y"]
                                + marker_mocap_data_trimmed[f"{left_mid_asis_mocap}-Y"]
                            )
                            / 2,
                            (
                                marker_mocap_data_trimmed[f"{right_mid_asis_mocap}-Z"]
                                + marker_mocap_data_trimmed[f"{left_mid_asis_mocap}-Z"]
                            )
                            / 2,
                        ]
                    ).T

                    # Calculate heading vectors
                    heading_vec_mono = mid_ASIS_mono - mid_PSIS_mono
                    heading_vec_mocap = mid_ASIS_mocap - mid_PSIS_mocap

                    # Project into the XZ plane (set Y to 0)
                    heading_vec_mono[:, 1] = 0
                    heading_vec_mocap[:, 1] = 0

                    # Normalize the vectors
                    heading_vec_mono_normalized = heading_vec_mono / np.linalg.norm(
                        heading_vec_mono, axis=1, keepdims=True
                    )
                    heading_vec_mocap_normalized = heading_vec_mocap / np.linalg.norm(
                        heading_vec_mocap, axis=1, keepdims=True
                    )

                    # Calculate the angular difference using the dot product
                    dot_products = np.einsum(
                        "ij,ij->i",
                        heading_vec_mono_normalized,
                        heading_vec_mocap_normalized,
                    )
                    angles = np.arccos(
                        np.clip(dot_products, -1.0, 1.0)
                    )  # Clip for numerical stability

                    # Convert to degrees if needed
                    angles_degrees = np.degrees(angles)

                    # Calculate the average angular difference
                    average_difference = np.mean(angles_degrees)

                    print(
                        f"Average Angular Difference: {average_difference:.2f} degrees"
                    )

                    # Rotate the mono marker data around the Y axis by the average angle in the XZ plane using rotate method
                    rotated_marker_video_data = rotate_dataframe(
                        shifted_video_data, "y", average_difference
                    )
                    # print(rotated_marker_data)

                    # Translation alignment
                    # Calculate the offset for the midpoints of the ASIS markers at timestep #1 of mocap
                    # Right knee

                    right_knee_mocap = "r_knee"
                    right_knee_mono = "r_knee"
                    r_knee_mocap = np.array(
                        [
                            marker_mocap_data_trimmed[f"{right_knee_mocap}-X"],
                            marker_mocap_data_trimmed[f"{right_knee_mocap}-Y"],
                            marker_mocap_data_trimmed[f"{right_knee_mocap}-Z"],
                        ]
                    ).T
                    r_knee_mono = np.array(
                        [
                            rotated_marker_video_data[f"{right_knee_mono}-X"],
                            rotated_marker_video_data[f"{right_knee_mono}-Y"],
                            rotated_marker_video_data[f"{right_knee_mono}-Z"],
                        ]
                    ).T

                    # Left knee
                    left_knee_mocap = "l_knee"
                    left_knee_mono = "l_knee"

                    l_knee_mocap = np.array(
                        [
                            marker_mocap_data_trimmed[f"{left_knee_mocap}-X"],
                            marker_mocap_data_trimmed[f"{left_knee_mocap}-Y"],
                            marker_mocap_data_trimmed[f"{left_knee_mocap}-Z"],
                        ]
                    ).T

                    l_knee_mono = np.array(
                        [
                            rotated_marker_video_data[f"{left_knee_mono}-X"],
                            rotated_marker_video_data[f"{left_knee_mono}-Y"],
                            rotated_marker_video_data[f"{left_knee_mono}-Z"],
                        ]
                    ).T

                    # Right ankle
                    right_ankle_mocap = "r_ankle"
                    right_ankle_mono = "r_ankle"

                    r_ankle_mocap = np.array(
                        [
                            marker_mocap_data_trimmed[f"{right_ankle_mocap}-X"],
                            marker_mocap_data_trimmed[f"{right_ankle_mocap}-Y"],
                            marker_mocap_data_trimmed[f"{right_ankle_mocap}-Z"],
                        ]
                    ).T

                    r_ankle_mono = np.array(
                        [
                            rotated_marker_video_data[f"{right_ankle_mono}-X"],
                            rotated_marker_video_data[f"{right_ankle_mono}-Y"],
                            rotated_marker_video_data[f"{right_ankle_mono}-Z"],
                        ]
                    ).T

                    # Left ankle
                    left_ankle_mocap = "l_ankle"
                    left_ankle_mono = "l_ankle"
                    l_ankle_mocap = np.array(
                        [
                            marker_mocap_data_trimmed[f"{left_ankle_mocap}-X"],
                            marker_mocap_data_trimmed[f"{left_ankle_mocap}-Y"],
                            marker_mocap_data_trimmed[f"{left_ankle_mocap}-Z"],
                        ]
                    ).T

                    l_ankle_mono = np.array(
                        [
                            rotated_marker_video_data[f"{left_ankle_mono}-X"],
                            rotated_marker_video_data[f"{left_ankle_mono}-Y"],
                            rotated_marker_video_data[f"{left_ankle_mono}-Z"],
                        ]
                    ).T

                    # Right shoulder
                    right_shoulder_mocap = "r_shoulder"
                    right_shoulder_mono = "r_shoulder"
                    r_shoulder_mocap = np.array(
                        [
                            marker_mocap_data_trimmed[f"{right_shoulder_mocap}-X"],
                            marker_mocap_data_trimmed[f"{right_shoulder_mocap}-Y"],
                            marker_mocap_data_trimmed[f"{right_shoulder_mocap}-Z"],
                        ]
                    ).T

                    r_shoulder_mono = np.array(
                        [
                            rotated_marker_video_data[f"{right_shoulder_mono}-X"],
                            rotated_marker_video_data[f"{right_shoulder_mono}-Y"],
                            rotated_marker_video_data[f"{right_shoulder_mono}-Z"],
                        ]
                    ).T

                    # Left shoulder
                    left_shoulder_mocap = "l_shoulder"
                    left_shoulder_mono = "l_shoulder"
                    l_shoulder_mocap = np.array(
                        [
                            marker_mocap_data_trimmed[f"{left_shoulder_mocap}-X"],
                            marker_mocap_data_trimmed[f"{left_shoulder_mocap}-Y"],
                            marker_mocap_data_trimmed[f"{left_shoulder_mocap}-Z"],
                        ]
                    ).T

                    l_shoulder_mono = np.array(
                        [
                            rotated_marker_video_data[f"{left_shoulder_mono}-X"],
                            rotated_marker_video_data[f"{left_shoulder_mono}-Y"],
                            rotated_marker_video_data[f"{left_shoulder_mono}-Z"],
                        ]
                    ).T

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
                    avg_x_offset = np.mean(
                        [
                            offset_x_asis,
                            offset_x_psis,
                            offset_x_r_knee,
                            offset_x_l_knee,
                            offset_x_r_ankle,
                            offset_x_l_ankle,
                            offset_x_r_shoulder,
                            offset_x_l_shoulder,
                        ]
                    )
                    avg_y_offset = np.mean(
                        [
                            offset_y_asis,
                            offset_y_psis,
                            offset_y_r_knee,
                            offset_y_l_knee,
                            offset_y_r_ankle,
                            offset_y_l_ankle,
                            offset_y_r_shoulder,
                            offset_y_l_shoulder,
                        ]
                    )
                    avg_z_offset = np.mean(
                        [
                            offset_z_asis,
                            offset_z_psis,
                            offset_z_r_knee,
                            offset_z_l_knee,
                            offset_z_r_ankle,
                            offset_z_l_ankle,
                            offset_z_r_shoulder,
                            offset_z_l_shoulder,
                        ]
                    )

                    print(f"Average X Offset: {avg_x_offset:.2f}")
                    print(f"Average Y Offset: {avg_y_offset:.2f}")
                    print(f"Average Z Offset: {avg_z_offset:.2f}")

                    # Apply the offset to all rotated markers of the video data (all columns ending with '-X', '-Y', '-Z')
                    for col in rotated_marker_video_data.columns:
                        if col.endswith("-X"):
                            rotated_marker_video_data[col] += avg_x_offset
                        elif col.endswith("-Y"):
                            rotated_marker_video_data[col] += avg_y_offset
                        elif col.endswith("-Z"):
                            rotated_marker_video_data[col] += avg_z_offset

                    # print(rotated_marker_video_data)

                    # Trim video markers data to match the mocap data if necessary (based on the Time column)

                    # Compute the mean per marker error for each marker. This is just the 2 norm /Euclidian distance. Compute the average error over all markers.
                    # first find the common markers using regex. e.g r-knee-X is the same as r.knee-X or knee.r-X
                    # then compute the error for each marker
                    marker_errors, average_error = compute_marker_errors(
                        rotated_marker_video_data, marker_mocap_data_trimmed
                    )
                    print(f"Marker Errors: {marker_errors}")
                    print(f"Average Error: {average_error:.2f}")

                    # export the marker errors to a csv
                    error_file_name = f"marker_errors_{movement}.csv"

                    error_markers_path_file = os.path.join(
                        error_markers_path, 'errors', error_file_name
                    )

                    if not os.path.exists(os.path.join(error_markers_path, 'errors')):
                        os.makedirs(os.path.join(error_markers_path, 'errors'))


                    with open(error_markers_path_file, "w") as file:
                        file.write("Marker,Error\n")
                        for marker, error in marker_errors.items():
                            file.write(f"{marker},{error}\n")
                        file.write(f"Average,{average_error}\n")

                    # plot the marker errors
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=list(marker_errors.keys()),
                            y=list(marker_errors.values()),
                            name="Marker Errors",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=list(marker_errors.keys()),
                            y=[average_error] * len(marker_errors),
                            mode="lines",
                            name="Average Error",
                        )
                    )
                    fig.update_layout(
                        title=f"Marker Errors for {movement}",
                        xaxis_title="Marker",
                        yaxis_title="Error",
                    )

                    # save the plot to a file
                    error_plot_file_name = f"marker_errors_plot_{movement}.html"


                    error_plot_file_path = os.path.join(
                        error_markers_path, 'errors', error_plot_file_name
                    )

                    fig.write_html(error_plot_file_path)


                    # TODO write the aligned mono data to file _sync.trc
                    synced_path = marker_video_path.replace(".trc", "_sync.trc")

                    write_trc(
                        keypoints3D=rotated_marker_video_data,
                        pathOutputFile=synced_path,
                        keypointNames=rotated_marker_video_data.columns[1:],
                        frameRate=100,
                    )

                    print("Wrote aligned mono data to file")


if __name__ == "__main__":
    main()
