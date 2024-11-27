from matplotlib import pyplot as plt
from utilsChecker import cross_corr_multiple_timeseries
import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
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
                mocap_time_start = marker_mocap_data['Time'][start]
                mocap_time_end = marker_mocap_data['Time'][end - 1]

                # Trim to match the length of mocap data
                shifted_video_data = shifted_video_data.iloc[:num_rows_mocap]

                # Extract time vectors
                time_mocap = marker_mocap_data['Time'].values
                time_video = shifted_video_data['Time'].values


                time_offset = mocap_time_start - time_video[0]

                # add the time offset to the time video
                time_video += time_offset


                # rotational alignment

                # convert all the data to float
                mocap_df = mocap_df.astype(float)
                mono_df = mono_df.astype(float)

                # Calculate midpoints for mono
                mid_PSIS_mono = np.array([
                    (mono_df['r_PSIS-X15'] + mono_df['l_PSIS-X16']) / 2,
                    (mono_df['r_PSIS-Y15'] + mono_df['l_PSIS-Y16']) / 2,
                    (mono_df['r_PSIS-Z15'] + mono_df['l_PSIS-Z16']) / 2
                ]).T

                mid_ASIS_mono = np.array([
                    (mono_df['r_ASIS-X13'] + mono_df['l_ASIS-X14']) / 2,
                    (mono_df['r_ASIS-Y13'] + mono_df['l_ASIS-Y14']) / 2,
                    (mono_df['r_ASIS-Z13'] + mono_df['l_ASIS-Z14']) / 2
                ]).T

                # Calculate midpoints for mocap
                mid_PSIS_mocap = np.array([
                    (mocap_df['r.PSIS-X28'] + mocap_df['L.PSIS-X30']) / 2,
                    (mocap_df['r.PSIS-Y28'] + mocap_df['L.PSIS-Y30']) / 2,
                    (mocap_df['r.PSIS-Z28'] + mocap_df['L.PSIS-Z30']) / 2
                ]).T

                mid_ASIS_mocap = np.array([
                    (mocap_df['r.ASIS-X27'] + mocap_df['L.ASIS-X29']) / 2,
                    (mocap_df['r.ASIS-Y27'] + mocap_df['L.ASIS-Y29']) / 2,
                    (mocap_df['r.ASIS-Z27'] + mocap_df['L.ASIS-Z29']) / 2
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

                # Rotate
                # the
                # mono
                # marker
                # data
                # based
                # on
                # this
                # average
                # angle(marker
                # rotation
                # function).You
                # now
                # have
                # temporally and rotationally
                # aligned
                # data, but
                # it
                # there’s
                # a
                # translational
                # offset.

                # TODO translation aligment

                # TODO write market errors to file

                # TODO write the aligned mono data to file _sync.trc

                # with open(lag_file_path, 'r') as file:
                #     lag = int(file.readline().split(':')[-1])
                #     correlation = float(file.readline().split(':')[-1])
                #
                # num_rows_mocap = marker_mocap_data.shape[0]
                # num_rows_video = marker_video_data.shape[0]
                #
                # shifted_marker_video_data = pd.DataFrame(columns=marker_video_data.columns, index=range(num_rows_mocap))
                #
                #
                # # TODO where to stop the slice since the signals are taken at different Hz
                # if lag > 0:
                #     start = lag
                #     end = min(num_rows_video + lag, num_rows_mocap)
                #     source_slice = marker_video_data.iloc[:end - lag]
                # elif lag < 0:
                #     start = 0
                #     end = min(num_rows_video, num_rows_mocap + lag)
                #     source_slice = marker_video_data.iloc[-lag:end - lag]
                # else:
                #     start = 0
                #     end = num_rows_video
                #     source_slice = marker_video_data.iloc[:end]
                #
                #
                # cleaned_shifted_marker_video_data = shifted_marker_video_data.dropna(how='all')
                #
                # time_mocap = marker_mocap_data['Time'].values
                # time_video = shifted_marker_video_data['Time'].values
                #
                #
                # # interpolate the cleaned_shifted_marker_video_data using the time_video and time_mocap
                # f = interp1d(time_video, shifted_marker_video_data.values[:, 2:], axis=0)
                # interpolated_shifted_marker_video_data = f(time_mocap)
                # interpolated_shifted_marker_video_data = pd.DataFrame(
                #     interpolated_shifted_marker_video_data,
                #     columns=shifted_marker_video_data.columns[2:]
                # )
                # interpolated_shifted_marker_video_data.insert(0, 'Time', time_mocap)
                # print(interpolated_shifted_marker_video_data)


# from matplotlib import pyplot as plt
#
# from utilsChecker import cross_corr_multiple_timeseries
# import os
# import pandas as pd
# import numpy as np
# from scipy.interpolate import interp1d
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
#
#
#
# def read_trc_file(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     # Find the line where the data header starts
#     start_line = 0
#     for i, line in enumerate(lines):
#         if line.startswith("Frame#"):
#             start_line = i
#             break
#
#     # Read the data from the identified starting line
#     data = pd.read_csv(
#         file_path,
#         delimiter='\t',
#         skiprows=start_line,
#         # header=[0, 1],  # Specify double header
#         engine='python'
#     )
#
#     # Replace the columns names containing "Unnamed : number" with the previous column name if it exists
#     for i, col in enumerate(data.columns):
#         if 'Unnamed' in col:
#             data.columns.values[i] = data.columns.values[i - 1]
#
#     data.columns = data.columns + '-' + data.iloc[0]
#     # set the first column name as Frame#
#     data.columns.values[0] = 'Frame#'
#     data.columns.values[1] = 'Time'
#
#     # drop the first row
#     data = data.drop(0)
#
#     return data
#
#
# repo_path = '/home/selim/opencap-mono'
# validation_videos_path = os.path.join(repo_path, 'LabValidation_withVideos1')
# output_path = os.path.join(repo_path, 'output')
#
# subjects_dirs = os.listdir(output_path)
#
# for subject in subjects_dirs:
#     MarkerDataDir = os.path.join(validation_videos_path, subject, 'MarkerData', 'Mocap')
#     subject_path = os.path.join(output_path, subject)
#     sessions = os.listdir(subject_path)
#
#     for session in sessions:
#         cameras = os.listdir(os.path.join(subject_path, session))
#
#         for camera in cameras:
#             movements = os.listdir(os.path.join(subject_path, session, camera))
#
#             for movement in movements:
#                 movement_path = os.path.join(subject_path, session, camera, movement, movement)
#                 marker_video_path = os.path.join(movement_path, 'MarkerData')
#                 if not os.path.exists(marker_video_path):
#                     continue
#                 # get the folder in marker_video_path
#                 marker_video_path = os.path.join(marker_video_path, os.listdir(marker_video_path)[0])
#                 # get the file path in marker_video_path with the .trc extension
#                 marker_video_path = os.path.join(marker_video_path, [f for f in os.listdir(marker_video_path) if f.endswith('.trc')][0])
#                 if not os.path.exists(marker_video_path):
#                     continue
#
#                 movement_file_name_trc = movement + '.trc'
#                 marker_mocap_path = os.path.join(MarkerDataDir, movement_file_name_trc)
#
#                 try:
#                     marker_video_data = read_trc_file(marker_video_path)
#                     marker_mocap_data = read_trc_file(marker_mocap_path)
#                 except Exception as e:
#                     print(f'Error: {e}')
#                     continue
#                 lag_file_name = f"lag_correlation_{movement}.txt" # lag_correlation_{video_file}.txt
#                 lag_file_path = os.path.join(movement_path, 'OpenSim', 'IK', 'shiftedIK', lag_file_name)
#                 if not os.path.exists(lag_file_path):
#                     continue
#                 with open(lag_file_path, 'r') as file:
#                     lag = int(file.readline().split(':')[-1])
#                     correlation = float(file.readline().split(':')[-1])
#
#                 # get num of rows mocap
#                 num_rows_mocap = marker_mocap_data.shape[0]
#                 # get num of rows video
#                 num_rows_video = marker_video_data.shape[0]
#
#                 # create a df with num rows = num rows mocap and the columns of the video data
#                 shifted_marker_video_data = pd.DataFrame(columns=marker_video_data.columns, index=range(num_rows_mocap))
#
#                 # Copy the values of the video data to the new DataFrame using the lag
#                 if lag > 0:
#                     # Positive lag: Shift the data forward
#                     start = lag
#                     end = min(num_rows_video + lag, num_rows_mocap)
#                     shifted_marker_video_data.iloc[start:end] = marker_video_data.iloc[:end - lag].values
#                 elif lag < 0:
#                     # Negative lag: Shift the data backward
#                     start = 0
#                     end = min(num_rows_video, num_rows_mocap + lag)
#                     shifted_marker_video_data.iloc[start:end] = marker_video_data.iloc[-lag:end - lag].values
#                 else:
#                     # No lag: Copy the data directly
#                     shifted_marker_video_data.iloc[:num_rows_video] = marker_video_data.values
#
#                 # shifted num of rows video
#                 num_rows_video = shifted_marker_video_data.shape[0]
#                 assert num_rows_video == num_rows_mocap, "The number of rows of the shifted video data should be equal to the number of rows of the mocap data."
#
#                 # TODO interpolate so the mono time vector is the same as the mocap time vector.
#                 from scipy.interpolate import interp1d
#                 import pandas as pd
#
#                 # Clean shifted_marker_video_data to remove rows with all NaN values
#                 cleaned_shifted_marker_video_data = shifted_marker_video_data.dropna(how='all')
#
#                 # Extract time vectors
#                 time_mocap = marker_mocap_data['Time'].values
#                 time_video = cleaned_shifted_marker_video_data['Time'].values
#
#                 # Adjust video time to align with mocap time
#                 time_offset = time_mocap[0] - time_video[0]
#                 adjusted_time_video = time_video + time_offset
#
#                 # Interpolate the marker data (skipping the Time and Frame# columns)
#                 f = interp1d(adjusted_time_video, cleaned_shifted_marker_video_data.values[:, 2:], axis=0,
#                              bounds_error=False, fill_value="extrapolate")
#
#                 # Perform the interpolation
#                 interpolated_shifted_marker_video_data = f(time_mocap)
#
#                 # Create a DataFrame with the interpolated data
#                 interpolated_shifted_marker_video_data = pd.DataFrame(
#                     interpolated_shifted_marker_video_data,
#                     columns=cleaned_shifted_marker_video_data.columns[2:]  # Exclude Time and Frame# from columns
#                 )
#
#                 # Add the Time column from mocap for reference
#                 interpolated_shifted_marker_video_data.insert(0, 'Time', time_mocap)
#                 print(interpolated_shifted_marker_video_data)

     # # Interpolate marker video data to match mocap time vector
                # interpolated_values = interp1d(
                #     time_video,
                #     shifted_video_data.iloc[:, 2:].values,  # Exclude non-marker columns (e.g., 'Frame#', 'Time')
                #     axis=0,
                #     bounds_error=False,
                #     fill_value="extrapolate"
                # )(time_mocap)
                #
                # # Prepare DataFrames with column names preserved
                # mocap_df = marker_mocap_data.iloc[:, 2:].copy()
                # mono_df = pd.DataFrame(interpolated_values, columns=shifted_video_data.columns[2:])
                #
                # # Add Time column back for clarity
                # mocap_df.insert(0, 'Time', time_mocap)
                # mono_df.insert(0, 'Time', time_mocap)
                #
                # # Ensure DataFrames have the same shape
                # assert mocap_df.shape[0] == mono_df.shape[0], "Shape mismatch between mocap and mono DataFrames."
                #
                # # Output for verification
                # print("Aligned Mocap DataFrame:")
                # print(mocap_df.head())
                # print("\nAligned Mono DataFrame:")
                # print(mono_df.head())
                #
                # # Save aligned data if needed
                # mocap_df.to_csv('aligned_mocap.csv', index=False)
                # mono_df.to_csv('aligned_mono.csv', index=False)

#
#                 # TODO rotational aligment
#
#                 # TODO translation aligment
#
#                 # TODO write market errors to file
#
#                 # TODO write the aligned mono data to file _sync.trc
