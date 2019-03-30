# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2
import numpy as np
import socket
import json
import pickle
#set up UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, # Internet
	socket.SOCK_DGRAM) # UDP
# construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--shape-predictor", required=True,
#	help="path to facial landmark predictor")
#args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./68.dat")

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=1).start()
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
#time.sleep(2.0)
K = [6.5308391993466671e+002, 0.0, 3.1950000000000000e+002,
     0.0, 6.5308391993466671e+002, 2.3950000000000000e+002,
     0.0, 0.0, 1.0]
D = [7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)

object_pts = np.float32([[6.825897, 6.760612, 4.402142],
                         [1.330353, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.311432, 5.485328, 3.987654],
                         [1.789930, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.005628, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652],
                         [2.774015, -2.080775, 5.048531],
                         [-2.774015, -2.080775, 5.048531],
                         [0.000000, -3.116408, 6.097667],
                         [0.000000, -7.415691, 4.070434]])

reprojectsrc = np.float32([[10.0, 10.0, 10.0],
                           [10.0, 10.0, -10.0],
                           [10.0, -10.0, -10.0],
                           [10.0, -10.0, 10.0],
                           [-10.0, 10.0, 10.0],
                           [-10.0, 10.0, -10.0],
                           [-10.0, -10.0, -10.0],
                           [-10.0, -10.0, 10.0]])

line_pairs = [[0, 1], [1, 2], [2, 3], [3, 0],
              [4, 5], [5, 6], [6, 7], [7, 4],
              [0, 4], [1, 5], [2, 6], [3, 7]]

#Head pose detection 
def get_head_pose(shape):
    image_pts = np.float32([shape[17], shape[21], shape[22], shape[26], shape[36],
                            shape[39], shape[42], shape[45], shape[31], shape[35],
                            shape[48], shape[54], shape[57], shape[8]])

    _, rotation_vec, translation_vec = cv2.solvePnP(object_pts, image_pts, cam_matrix, dist_coeffs)

    reprojectdst, _ = cv2.projectPoints(reprojectsrc, rotation_vec, translation_vec, cam_matrix,
                                        dist_coeffs)

    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))

    # calc euler angle
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    return reprojectdst, euler_angle
# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	rects = detector(gray, 0)
	image = np.zeros((400,400,3), np.uint8)
	# check to see if a face was detected, and if so, draw the total
	# number of faces on the frame
	if len(rects) > 0:
		text = "{} face(s) found".format(len(rects))
		cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, (0, 0, 255), 2)
		# loop over the face detections
	for rect in rects:
		
		# compute the bounding box of the face and draw it on the
		# frame
		(bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
			(0, 255, 0), 1)
 
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		data = shape
		print tuple(shape[52])
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw each of them
		for (i, (x, y)) in enumerate(shape):
			
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
			cv2.putText(frame, str(i + 1), (x - 10, y - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
			data[i] = tuple(shape[i])
		P = enumerate(shape)
		image = np.zeros((400,400,3), np.uint8)
	# chin
		for (i) in range(1,17):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 2)
			
	# left eyebrow
		for (i) in range(18,22):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 2)
			
	# right eyebrow
		for (i) in range(23,27):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 2)
			
	# nose ridge
		for (i) in range(28,31):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 1)
			
	# bottom of nose
		for (i) in range(32,36):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 1)
			
	#left eye
		for (i) in range(37,42):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 1)
		cv2.line(image, tuple(shape[37-1]), tuple(shape[42-1]), (0,255,0), 1)
	# right eye
		for (i) in range(43,48):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 1)
		cv2.line(image, tuple(shape[43-1]), tuple(shape[48-1]), (0,255,0), 1)
	# mouth Outer
		for (i) in range(49,60):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 1)
		cv2.line(image, tuple(shape[49-1]), tuple(shape[60-1]), (0,255,0), 1)
	#mouth innner
		for (i) in range(61,68):
			A1 = tuple(shape[i-1])
			B1 = tuple(shape[i])
			cv2.line(image, (A1), (B1), (0,255,0), 1)
		cv2.line(image, tuple(shape[61-1]), tuple(shape[68-1]), (0,255,0), 1)
		reprojectdst, euler_angle = get_head_pose(shape)
		for start, end in line_pairs:
			cv2.line(image, reprojectdst[start], reprojectdst[end], (0, 0, 255))

			cv2.putText(image, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
						0.75, (255, 0, 0), thickness=2)
			cv2.putText(image, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
						0.75, (255, 0, 0), thickness=2)
			cv2.putText(image, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
						0.75, (255, 0, 0), thickness=2)
		MESSAGE = json.dumps({"Face":data.tolist(),  "Pose":euler_angle.tolist()})
		sock.sendto(MESSAGE.encode(), (UDP_IP, UDP_PORT))
	# show the frame
	cv2.imshow("Frame", frame)
	#cv2.imshow("line",image)
	
	key = cv2.waitKey(1) & 0xFF
  
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
