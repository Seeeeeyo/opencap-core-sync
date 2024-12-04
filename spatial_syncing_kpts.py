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
from utils_trc import write_trc, TRCFile, transform_from_tuple_array, align_trc_files


def calculate_midpoint(trc_file, right_marker, left_marker):
    assert trc_file.marker_exists(right_marker), f"Marker {right_marker} does not exist in the TRC file."
    assert trc_file.marker_exists(left_marker), f"Marker {left_marker} does not exist in the TRC file."

    right_data = trc_file.marker(right_marker)
    left_data = trc_file.marker(left_marker)

    midpoint = np.array(
        [
            (right_data[:, 0] + left_data[:, 0]) / 2,
            (right_data[:, 1] + left_data[:, 1]) / 2,
            (right_data[:, 2] + left_data[:, 2]) / 2,
        ]
    ).T

    return midpoint

def main():
    repo_path = "/home/selim/opencap-mono"
    validation_videos_path = os.path.join(repo_path, "LabValidation_withVideos1")
    output_path = os.path.join(repo_path, "output")
    frame_rate = 100

    # subjects_dirs = os.listdir(output_path)


    # HARDCODED SUBJECTS
    subjects_dirs = ['subject2']

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
                        d for d in os.listdir(marker_video_path) if d != 'errors' and d != 'shiftedIK'
                    ]
                    if not marker_video_subdirs:
                        continue

                    print(f"Filepath: {marker_video_path}")

                    # get the folder in marker_video_subdirs list, it doesnt have any extension
                    folder = next((d for d in marker_video_subdirs if '.' not in d), None)

                    if folder is None:
                        continue

                    marker_video_path = os.path.join(marker_video_path, folder)
                    marker_video_files = [
                        f for f in os.listdir(marker_video_path) if f.endswith(".trc")
                    ]
                    if not marker_video_files:
                        continue

                    # if 'sync' is in one of the files, then continue
                    # if any("_sync.trc" in file for file in marker_video_files):
                    # continue
                        # delete the file

                    # if 'sync' is in one of the files, then delete it
                    for file in marker_video_files:
                        if "_sync.trc" in file:
                            os.remove(os.path.join(marker_video_path, file))


                    marker_video_path = os.path.join(marker_video_path, marker_video_files[0])
                    if not os.path.exists(marker_video_path):
                        continue

                    movement_file_name_trc = movement + ".trc"
                    marker_mocap_path = os.path.join(
                        MarkerDataDir, movement_file_name_trc
                    )

                    trc_mono = TRCFile(marker_video_path)
                    trc_mocap = TRCFile(marker_mocap_path)

                    mocap_start, mocap_end = trc_mocap.get_start_end_times()
                    mono_start, mono_end = trc_mono.get_start_end_times()

                    print(f"Mocap Start: {mocap_start}, Mocap End: {mocap_end}")
                    print(f"Mono Start: {mono_start}, Mono End: {mono_end}")


                    # mono_marker_names = extract_marker_names(marker_video_data)
                    # mocap_marker_names = extract_marker_names(marker_mocap_data)

                    lag_file_name = f"lag_correlation_{movement}.txt"
                    lag_file_path = os.path.join(
                        movement_path, "OpenSim", "IK", "shiftedIK", lag_file_name
                    )
                    if not os.path.exists(lag_file_path):
                        continue


                    with open(lag_file_path, "r") as file:
                        lag = int(file.readline().split(":")[-1])
                        correlation = float(file.readline().split(":")[-1])


                    if trc_mono.get_frequency() != frame_rate:
                        print(f"Resampling mono TRC to {frame_rate} Hz.")
                        trc_mono.resample_trc(target_frequency=frame_rate)

                    if trc_mocap.get_frequency() != frame_rate:
                        print(f"Resampling mocap TRC to {frame_rate} Hz.")
                        trc_mocap.resample_trc(target_frequency=frame_rate)


                    # Shift
                    align_trc_files(trc_mono, trc_mocap, lag)
                    # trc_mono.adjust_and_slice_by_lag(lag=lag,trc_mocap=trc_mocap)
                    # trc_mono.adjust_and_slice_by_lag(lag=lag, target_length=trc_mocap.data.shape[0])
                    print(f"Shifted mono TRC by {lag} frames.")

                    mocap_start, mocap_end = trc_mocap.get_start_end_times()
                    mono_start, mono_end = trc_mono.get_start_end_times()

                    print(f"Mocap Start: {mocap_start}, Mocap End: {mocap_end}")
                    print(f"Mono Start: {mono_start}, Mono End: {mono_end}")


                    # # TODO verify if the time offset is correct and if this is needed for trc
                    # time_offset = mocap_start - mono_start
                    # print(f"Time Offset: {time_offset}")
                    #
                    # # add the time offset to the time video
                    # trc_mono.add_time_offset(time_offset)
                    #
                    # # trim the video data to match the length of the mocap data
                    # trc_mono.trim_to_match(mocap_start, mocap_end)


                    trc_mono_marker_names = trc_mono.get_marker_names()
                    trc_mocap_marker_names = trc_mocap.get_marker_names()

                    # print(f"Mono Marker Names: {trc_mono_marker_names}")
                    # print(f"Mocap Marker Names: {trc_mocap_marker_names}")


                    # get the metric of the markers
                    mono_metric = trc_mono.get_metric_trc()
                    mocap_metric = trc_mocap.get_metric_trc()

                    # convert the markers to the same metric
                    if mono_metric != 'mm':
                        trc_mono.convert_to_metric_trc(current_metric=mono_metric, target_metric='mm')
                    if mocap_metric != 'mm':
                        trc_mocap.convert_to_metric_trc(current_metric=mocap_metric, target_metric='mm')


                    # Rotational alignment

                    trc_mocap_trimmed = trc_mocap.copy()
                    trc_mocap_trimmed.trim_to_match(mono_start, mono_end)

                    # Calculate midpoints for mono
                    mid_PSIS_mono = calculate_midpoint(trc_mono, "r_PSIS", "l_PSIS")
                    mid_ASIS_mono = calculate_midpoint(trc_mono, "r_ASIS", "l_ASIS")

                    # Calculate midpoints for mocap
                    mid_PSIS_mocap = calculate_midpoint(trc_mocap_trimmed, "r.PSIS", "L.PSIS")
                    mid_ASIS_mocap = calculate_midpoint(trc_mocap_trimmed, "r.ASIS", "L.ASIS")


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
                    trc_mono.rotate(axis="y", value=average_difference)



                    # Translation alignment
                    # Calculate the offset for the midpoints of the ASIS markers at timestep #1 of mocap
                    # Right knee

                    # Define marker names for joints
                    markers = {
                        "r_knee": "r_knee",
                        "l_knee": "L_knee",
                        "r_ankle": "r_ankle",
                        "l_ankle": "L_ankle",
                        "r_shoulder": "R_Shoulder",
                        "l_shoulder": "L_Shoulder",
                        "r_ASIS": "r.ASIS",
                        "l_ASIS": "L.ASIS",
                        "r_PSIS": "r.PSIS",
                        "l_PSIS": "L.PSIS"
                    }

                    # Collect offsets for all markers
                    offsets_x, offsets_y, offsets_z = [], [], []

                    # Loop through each marker to calculate offsets
                    for mono_marker, mocap_marker in markers.items():
                        assert trc_mono.marker_exists(
                            mono_marker), f"Marker {mono_marker} does not exist in the mono TRC file."
                        assert trc_mocap_trimmed.marker_exists(
                            mocap_marker), f"Marker {mocap_marker} does not exist in the mocap TRC file."

                        mono_data = trc_mono.marker(mono_marker)
                        mocap_data = trc_mocap_trimmed.marker(mocap_marker)

                        offsets_x.append(mocap_data[0, 0] - mono_data[0, 0])
                        offsets_y.append(mocap_data[0, 1] - mono_data[0, 1])
                        offsets_z.append(mocap_data[0, 2] - mono_data[0, 2])

                    # Calculate average offsets
                    avg_x_offset = np.mean(offsets_x)
                    avg_y_offset = np.mean(offsets_y)
                    avg_z_offset = np.mean(offsets_z)

                    # Print results
                    print(f"Average X Offset: {avg_x_offset:.2f} mm")
                    print(f"Average Y Offset: {avg_y_offset:.2f} mm")
                    print(f"Average Z Offset: {avg_z_offset:.2f} mm")

                    # Apply the offset to all rotated markers of the video data (all columns ending with '-X', '-Y', '-Z')
                    trc_mono.offset(axis='x', value=avg_x_offset)
                    trc_mono.offset(axis='y', value=avg_y_offset)
                    trc_mono.offset(axis='z', value=avg_z_offset)

                    # Compute the mean per marker error for each marker. This is just the 2 norm /Euclidian distance. Compute the average error over all markers.
                    # first find the common markers using regex. e.g r-knee-X is the same as r.knee-X or knee.r-X
                    # then compute the error for each marker

                    # marker_errors, average_error = compute_marker_errors(
                    #     rotated_marker_video_data, marker_mocap_data_trimmed
                    # )
                    # print(f"Marker Errors: {marker_errors}")
                    # print(f"Average Error: {average_error:.2f}")

                    # export the marker errors to a csv
                    # error_file_name = f"marker_errors_{movement}.csv"

                    # error_markers_path_file = os.path.join(
                    #     error_markers_path, 'errors', error_file_name
                    # )

                    # if not os.path.exists(os.path.join(error_markers_path, 'errors')):
                    #     os.makedirs(os.path.join(error_markers_path, 'errors'))


                    # with open(error_markers_path_file, "w") as file:
                    #     file.write("Marker,Error\n")
                    #     for marker, error in marker_errors.items():
                    #         file.write(f"{marker},{error}\n")
                    #     file.write(f"Average,{average_error}\n")
                    #
                    # # plot the marker errors
                    # fig = go.Figure()
                    # fig.add_trace(
                    #     go.Bar(
                    #         x=list(marker_errors.keys()),
                    #         y=list(marker_errors.values()),
                    #         name="Marker Errors (mm)",
                    #     )
                    # )
                    # fig.add_trace(
                    #     go.Scatter(
                    #         x=list(marker_errors.keys()),
                    #         y=[average_error] * len(marker_errors),
                    #         mode="lines",
                    #         name="Average Error (mm)",
                    #     )
                    # )
                    # fig.update_layout(
                    #     title=f"Marker Errors for {movement} (mm)",
                    #     xaxis_title="Marker",
                    #     yaxis_title="Error (mm)",
                    # )
                    #
                    # # save the plot to a file
                    # error_plot_file_name = f"marker_errors_plot_{movement}.html"
                    #
                    #
                    # error_plot_file_path = os.path.join(
                    #     error_markers_path, 'errors', error_plot_file_name
                    # )
                    #
                    # fig.write_html(error_plot_file_path)



                    # # rotated_marker_video_data = rotated_marker_video_data.drop(columns=["Time"])
                    # # get the sets of markers without the suffixes -X, -Y, -Z
                    # base_markers = set()
                    # for col in rotated_marker_video_data.columns:
                    #     if col in ["Time", "Frame#"]:
                    #         continue
                    #     col_base = col[:-2]
                    #     if len(col_base) > 1:
                    #         base_markers.add(col_base)
                    #
                    # # reshape the video marker data to (num_frames, num_markers * 3)
                    # # markers to write is a np array of shape (num_frames, num_markers * 3)
                    # num_frames = rotated_marker_video_data.shape[0]
                    # num_markers = len(base_markers)
                    # markers_to_write = np.zeros((num_frames, num_markers * 3))
                    # marker_names_to_write = []
                    #
                    # for i, marker in enumerate(base_markers):
                    #     x = rotated_marker_video_data[f"{marker}-X"].values
                    #     y = rotated_marker_video_data[f"{marker}-Y"].values
                    #     z = rotated_marker_video_data[f"{marker}-Z"].values
                    #     markers_to_write[:, i * 3] = x
                    #     markers_to_write[:, i * 3 + 1] = y
                    #     markers_to_write[:, i * 3 + 2] = z
                    #     marker_names_to_write.append(marker)

                    # TODO write the aligned mono data to file _sync.trc
                    synced_path = marker_video_path.replace(".trc", "_sync.trc")

                    trc_mono.convert_to_metric_trc(current_metric='mm', target_metric='m')
                    start_time, end_time = trc_mono.get_start_end_times()

                    print(f"Synced Mono Start Time: {start_time}")
                    print(f"Synced Mono End Time: {end_time}")

                    mono_synced = transform_from_tuple_array(trc_mono.data)

                    write_trc(
                        keypoints3D=mono_synced,
                        pathOutputFile=synced_path,
                        keypointNames=trc_mono_marker_names,
                        frameRate=frame_rate,
                        t_start=start_time,
                    )

                    # load the new trc file to ensure the data was written correctly
                    trc_synced = TRCFile(synced_path)
                    trc_synced_marker_names = trc_synced.get_marker_names()

                    print(f"Wrote synced marker data to: {synced_path}")
                    print("-" * 50)


if __name__ == "__main__":
    main()
