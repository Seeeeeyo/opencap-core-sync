from utilsChecker import cross_corr_multiple_timeseries
import os


def read_mot_file(file_path):
    # TODO function to read the mot file

    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def cross_correlate():
    name = "squats1.mot"
    mocap_ik_path = "/home/selim/opencap-mono/LabValidation_withVideos1/subject2/OpenSimData/Mocap/IK"
    mocap_ik_path = os.path.join(mocap_ik_path, name)

    video_ik_path = "/home/selim/opencap-mono/LabValidation_withVideos1/subject2/OpenSimData/Video/OpenPose_highAccuracy/5-cameras/IK"
    video_ik_path = os.path.join(video_ik_path, name)


    # read the files
    mocap_data = read_mot_file(mocap_ik_path)
    video_data = read_mot_file(video_ik_path)

    # process the data
    # TODO process the data


    # run the cross correlation
    cross_corr_multiple_timeseries(mocap_ik_path, video_ik_path)


if __name__ == "__main__":
    cross_correlate()


