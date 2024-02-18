import math
import cv2
import numpy as np
import json
import threading
import socket
import time
from pathlib import Path
from pyquaternion import Quaternion
from cv2 import aruco


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert isRotationMatrix(R)

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


class Detector:
    def __init__(self, detector, cam_matrix, cam_coeff, tag_size):
        self.detector = detector
        self.cam_matrix = cam_matrix
        self.cam_coeff = cam_coeff
        self.tag_size = tag_size

        self.tags = []

    def detect(self, frame):
        self.tags.clear()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (corners, ids, rejected) = self.detector.detectMarkers(gray)

        if ids is not None and len(ids) >= 0:
            rvecs, tvecs, points = aruco.estimatePoseSingleMarkers(
                corners, self.tag_size, self.cam_matrix, self.cam_coeff
            )
            for i in range(len(tvecs)):
                rvec, tvec = rvecs[i][0], tvecs[i][0]

                matrix, _ = cv2.Rodrigues(rvec)
                euler = rotationMatrixToEulerAngles(matrix)
                quat = Quaternion(matrix=matrix)
                relative = quat.inverse.rotate(tvec)

                tag_data = {
                    "id": int(ids[i][0]),
                    "relative": relative.tolist(),
                    "euler": euler.tolist(),
                }

                self.tags.append(tag_data)
                cv2.drawFrameAxes(
                    frame, cam_matrix, cam_coeff, rvec, tvec, self.tag_size * 1.1
                )

        frame = aruco.drawDetectedMarkers(frame, corners, ids, borderColor=(0, 255, 0))
        return frame

    def detect_loop(self):
        while True:
            self.detect()

    def start_thread(self):
        self.thr = threading.Thread(target=self.detect_loop, daemon=True)
        self.thr.start()

    def get_data_string(self):
        tag_str = json.dumps(self.tags)
        return tag_str


class UDP_Sender:
    def __init__(self, ip, port):
        self.port = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.portnum = port
        self.remote = ip

    def send(self, data):
        self.port.sendto(bytes(data, "utf-8"), (self.remote, self.portnum))


marker_size = 0.005 # Meters
vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("Cannot open camera")
    exit()

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)

path = Path(__file__).parent
cam_matrix = np.loadtxt(path / "camera_matrix.txt")
cam_coeff = np.loadtxt(path / "camera_coeff.txt")

board = aruco.GridBoard(
    size=(2, 2), markerLength=50, markerSeparation=5, dictionary=aruco_dict
)

aruco_detector = aruco.ArucoDetector(dictionary=aruco_dict)

detector = Detector(aruco_detector, cam_matrix, cam_coeff, marker_size)
sender = UDP_Sender("10.77.42.2", 11753)
rate = 30

writer = cv2.VideoWriter(
    "out_video.mp4", cv2.VideoWriter_fourcc(*"DIVX"), rate, (640, 480)
)

if __name__ == "__main__":
    try:
        while True:
            start_ns = time.time_ns()

            ret, frame = vc.read()
            if not ret:
                print("Can't receive frame (stream end?)")
                pass

            frame = detector.detect(frame)

            resized = cv2.resize(frame, (640, 480))
            resized = cv2.putText(
                resized,
                f"7742_ArUco_CPS Time:{time.time()}",
                (5, 475),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 0, 255),
                1,
            )
            writer.write(resized)

            data_str = detector.get_data_string()
            if data_str == None or data_str == "":
                continue

            print(data_str)
            sender.send(data_str)

            end_ns = time.time_ns()
            rate_ns = 1000000000 / rate
            delta_ns = end_ns - start_ns
            wait_sec = max((rate_ns - delta_ns) / 1000000000, 0)
            time.sleep(wait_sec)
    finally:
        print("Stopping!")

        vc.release()
        writer.release()
