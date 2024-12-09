
import numpy as np
import plotly.graph_objects as go
from utils_trc import write_trc, TRCFile, transform_from_tuple_array, align_trc_files
import os
from loguru import logger

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

    subjects_dirs = os.listdir(output_path)

    run_ik = True
    single_run = True

    # HARDCODED SUBJECTS
    if single_run:
        hd_subjects_dirs = ['subject3']
        hd_sessions_dirs = ['Session1']
        hd_cameras = ['Cam1']
        hd_movements = ['walking3']
        logger.info("Running on a single subject.")

    if single_run:
        subjects_dirs = hd_subjects_dirs

    for subject in subjects_dirs:

        MarkerDataDir = os.path.join(
            validation_videos_path, subject, "MarkerData", "Mocap"
        )
        subject_path = os.path.join(output_path, subject)

        if single_run:
            sessions = hd_sessions_dirs
        else:
            sessions = os.listdir(subject_path)

        for session in sessions:
            if single_run:
                cameras = hd_cameras
            else:
                cameras = os.listdir(os.path.join(subject_path, session))

            for camera in cameras:
                if single_run:
                    movements = hd_movements
                else:
                    movements = os.listdir(os.path.join(subject_path, session, camera))

                for movement in movements:
                    movement_folder = os.path.join(
                        subject_path, session, camera, movement
                    )
                    # list the folders in movement_folder
                    movement_folders = os.listdir(movement_folder)

                    # if the movement folder is empty, continue
                    if not movement_folders:
                        logger.info(f"Skipping {movement_folder} as it is empty.")
                        continue

                    # if there are multiple folders in the movement folder, get the one with 'trimmed' in the name
                    if len(movement_folders) > 1:
                        trimmed = next((d for d in movement_folders if 'trimmed' in d), None)
                        if movement_folder is None:
                            logger.info(f"Skipping {movement_folder} as it is empty.")
                            # continue
                        movement_path = os.path.join(movement_folder, trimmed)
                    else:
                        movement_path = os.path.join(movement_folder, movement_folders[0])


                    marker_video_path = os.path.join(movement_path, "MarkerData")
                    error_markers_path = os.path.join(movement_path, "MarkerData")
                    if not os.path.exists(marker_video_path):
                        continue

                    marker_video_subdirs = [
                        d for d in os.listdir(marker_video_path) if d != 'errors' and d != 'shiftedIK'
                    ]
                    if not marker_video_subdirs:
                        continue

                    logger.info(f"Filepath: {marker_video_path}")

                    # get the folder in marker_video_subdirs list, it doesnt have any extension
                    # folder = next((d for d in marker_video_subdirs if '.' not in d), None)
                    folder = next((d for d in marker_video_subdirs if '.' not in d and 'wham_result' not in d), None)

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

                    logger.info(f"Mocap Start: {mocap_start}, Mocap End: {mocap_end}")
                    logger.info(f"Mono Start: {mono_start}, Mono End: {mono_end}")


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
                        logger.info(f"Resampling mono TRC to {frame_rate} Hz.")
                        trc_mono.resample_trc(target_frequency=frame_rate)

                    if trc_mocap.get_frequency() != frame_rate:
                        logger.info(f"Resampling mocap TRC to {frame_rate} Hz.")
                        trc_mocap.resample_trc(target_frequency=frame_rate)


                    # Shift
                    align_trc_files(trc_mono, trc_mocap, lag)
                    # trc_mono.adjust_and_slice_by_lag(lag=lag,trc_mocap=trc_mocap)
                    # trc_mono.adjust_and_slice_by_lag(lag=lag, target_length=trc_mocap.data.shape[0])
                    logger.info(f"Shifted mono TRC by {lag} frames.")

                    mocap_start, mocap_end = trc_mocap.get_start_end_times()
                    mono_start, mono_end = trc_mono.get_start_end_times()

                    logger.info(f"Mocap Start: {mocap_start}, Mocap End: {mocap_end}")
                    logger.info(f"Mono Start: {mono_start}, Mono End: {mono_end}")


                    # # TODO verify if the time offset is correct and if this is needed for trc
                    # time_offset = mocap_start - mono_start
                    # logger.info(f"Time Offset: {time_offset}")
                    #
                    # # add the time offset to the time video
                    # trc_mono.add_time_offset(time_offset)
                    #
                    # # trim the video data to match the length of the mocap data
                    # trc_mono.trim_to_match(mocap_start, mocap_end)


                    trc_mono_marker_names = trc_mono.get_marker_names()
                    trc_mocap_marker_names = trc_mocap.get_marker_names()

                    # logger.info(f"Mono Marker Names: {trc_mono_marker_names}")
                    # logger.info(f"Mocap Marker Names: {trc_mocap_marker_names}")


                    # get the metric of the markers
                    mono_metric = trc_mono.get_metric_trc()
                    mocap_metric = trc_mocap.get_metric_trc()

                    # convert the markers to the same metric
                    if mono_metric != 'mm':
                        trc_mono.convert_to_metric_trc(current_metric=mono_metric, target_metric='mm')
                        logger.info(f"Converted mono markers from {mono_metric} to mm.")
                    if mocap_metric != 'mm':
                        trc_mocap.convert_to_metric_trc(current_metric=mocap_metric, target_metric='mm')
                        logger.info(f"Converted mocap markers from {mocap_metric} to mm.")


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

                    logger.info(
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

                    # if 'walking' in movement then take the hal length index to calculate the offset
                    idx = 0
                    
                    # Loop through each marker to calculate offsets
                    for mono_marker, mocap_marker in markers.items():
                        assert trc_mono.marker_exists(
                            mono_marker), f"Marker {mono_marker} does not exist in the mono TRC file."
                        assert trc_mocap_trimmed.marker_exists(
                            mocap_marker), f"Marker {mocap_marker} does not exist in the mocap TRC file."

                        mono_data = trc_mono.marker(mono_marker)
                        mocap_data = trc_mocap_trimmed.marker(mocap_marker)

                        offsets_x.append(mocap_data[idx, 0] - mono_data[0, 0])
                        offsets_y.append(mocap_data[idx, 1] - mono_data[0, 1])
                        offsets_z.append(mocap_data[idx, 2] - mono_data[0, 2])


                    # Calculate average offsets
                    avg_x_offset = np.mean(offsets_x)
                    avg_y_offset = np.mean(offsets_y)
                    avg_z_offset = np.mean(offsets_z)

                    # Print results
                    logger.info(f"Average X Offset: {avg_x_offset:.2f} mm")
                    logger.info(f"Average Y Offset: {avg_y_offset:.2f} mm")
                    logger.info(f"Average Z Offset: {avg_z_offset:.2f} mm")

                    # Apply the offset to all rotated markers of the video data (all columns ending with '-X', '-Y', '-Z')
                    trc_mono.offset(axis='x', value=avg_x_offset)
                    trc_mono.offset(axis='y', value=avg_y_offset)
                    trc_mono.offset(axis='z', value=avg_z_offset)

                    # Compute the mean per marker error for each marker. This is just the 2 norm /Euclidian distance. Compute the average error over all markers.
                    # first find the common markers using regex. e.g r-knee-X is the same as r.knee-X or knee.r-X
                    # then compute the error for each marker

                    # compute the marker error for the shoulders at the first frame in the z axis
                    right_shoulder_error = trc_mono.marker("r_shoulder")[0, 2] - trc_mocap_trimmed.marker("R_Shoulder")[0, 2]
                    logger.info(f"Right shoulder Error: {right_shoulder_error:.2f} mm")
                    left_shoulder_error = trc_mono.marker("l_shoulder")[0, 2] - trc_mocap_trimmed.marker("L_Shoulder")[0, 2]
                    logger.info(f"Left shoulder Error: {left_shoulder_error:.2f} mm")

                    # print the z distance between left and right shoulders
                    shoulder_distance_mono = np.abs(trc_mono.marker("r_shoulder")[0, 2] - trc_mono.marker("l_shoulder")[0, 2])
                    shoulder_distance_mocap = np.abs(trc_mocap_trimmed.marker("R_Shoulder")[0, 2] - trc_mocap_trimmed.marker("L_Shoulder")[0, 2])
                    logger.info(f"Shoulder Distance Mono: {shoulder_distance_mono:.2f} mm")
                    logger.info(f"Shoulder Distance Mocap: {shoulder_distance_mocap:.2f} mm")
                    # print the difference between the two distances
                    shoulder_distance_error = shoulder_distance_mocap - shoulder_distance_mono
                    logger.info(f"Shoulder Distance Error: {shoulder_distance_error:.2f} mm")

                    # apply offset to the mono trc file for the shoulders
                    # trc_mono.offset(axis='z', value=right_shoulder_error, single_marker='r_shoulder')
                    # trc_mono.offset(axis='z', value=left_shoulder_error, single_marker='l_shoulder')
                    # logger.info(f"Applied offset to the mono trc file for the shoulders.")


                    # Compute the mean per marker error for each marker
                    marker_errors = {}
                    for mono_marker, mocap_marker in markers.items():
                        assert trc_mono.marker_exists(
                            mono_marker), f"Marker {mono_marker} does not exist in the mono TRC file."
                        assert trc_mocap_trimmed.marker_exists(
                            mocap_marker), f"Marker {mocap_marker} does not exist in the mocap TRC file."

                        mono_data = trc_mono.marker(mono_marker)
                        mocap_data = trc_mocap_trimmed.marker(mocap_marker)

                        # Compute the Euclidean distance (2-norm) for each frame
                        errors = np.linalg.norm(mono_data - mocap_data, axis=1)
                        marker_errors[mono_marker] = np.mean(errors)

                    # Compute the average error over all markers
                    average_error = np.mean(list(marker_errors.values()))

                    logger.info(f"Marker Errors: {marker_errors}")
                    logger.info(f"Average Error: {average_error:.2f}")

                    # Export the marker errors to a CSV
                    error_file_name = f"marker_errors_{movement}.csv"
                    error_markers_path_file = os.path.join(error_markers_path, 'errors', error_file_name)

                    if not os.path.exists(os.path.join(error_markers_path, 'errors')):
                        os.makedirs(os.path.join(error_markers_path, 'errors'))

                    with open(error_markers_path_file, "w") as file:
                        file.write("Marker,Error\n")
                        for marker, error in marker_errors.items():
                            file.write(f"{marker},{error}\n")
                        file.write(f"Average,{average_error}\n")

                    # Plot the marker errors
                    fig = go.Figure()
                    fig.add_trace(
                        go.Bar(
                            x=list(marker_errors.keys()),
                            y=list(marker_errors.values()),
                            name="Marker Errors (mm)",
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=list(marker_errors.keys()),
                            y=[average_error] * len(marker_errors),
                            mode="lines",
                            name="Average Error (mm)",
                        )
                    )
                    fig.update_layout(
                        title=f"Marker Errors for {movement} (mm)",
                        xaxis_title="Marker",
                        yaxis_title="Error (mm)",
                    )

                    # Save the plot to a file
                    error_plot_file_name = f"marker_errors_plot_{movement}.html"
                    error_plot_file_path = os.path.join(error_markers_path, 'errors', error_plot_file_name)
                    fig.write_html(error_plot_file_path)

                    synced_path = marker_video_path.replace(".trc", "_sync.trc")

                    # if the original trc mocap file is in meters, convert the mono trc file to meters to be able to visualize them together in opensim
                    # if mocap_metric == 'm':
                    #     trc_mono.convert_to_metric_trc(current_metric='mm', target_metric='m')
                    #     logger.info(f"Converted synced mono markers from mm to m to match mocap markers unit.")

                    start_time, end_time = trc_mono.get_start_end_times()

                    logger.info(f"Synced Mono Start Time: {start_time}")
                    logger.info(f"Synced Mono End Time: {end_time}")

                    mono_synced = transform_from_tuple_array(trc_mono.data)

                    write_trc(
                        keypoints3D=mono_synced,
                        pathOutputFile=synced_path,
                        keypointNames=trc_mono_marker_names,
                        frameRate=frame_rate,
                        unit='mm',
                        t_start=start_time,
                    )

                    # load the new trc file to ensure the data was written correctly
                    trc_synced = TRCFile(synced_path)
                    trc_synced_marker_names = trc_synced.get_marker_names()

                    logger.info(f"Wrote synced marker data to: {synced_path}")

                    if run_ik:
                        # Run IK on the synced data
                        from utilsOpenSim import runIKTool
                        # TODO use a model without patella to run IK faster for now
                        pathOutputMotion = runIKTool(pathGenericSetupFile='/home/selim/opencap-mono/utils/opensim/IK/Setup_IK_SMPL.xml',
                                  pathScaledModel=os.path.join(movement_path, "OpenSim", "Model", folder, "LaiUhlrich2022_withMarkers_scaled_no_patella.osim"),
                                  pathTRCFile= synced_path,
                                  pathOutputFolder=os.path.join(movement_path, "OpenSim", "IK", "shiftedIK"))

                        logger.info(f"Ran IK on synced data. Results saved to: {pathOutputMotion}")
                    logger.info("-" * 70)


if __name__ == "__main__":
    main()
