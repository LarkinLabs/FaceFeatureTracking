import socket
import cv2
import numpy as np
import json
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

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
sock = socket.socket(socket.AF_INET, # Internet
	socket.SOCK_DGRAM) # UDP
sock.bind((UDP_IP, UDP_PORT))

while True:
	message, addr = sock.recvfrom(102400) # buffer size is 1024 bytes
	data = json.loads(message.decode())
	shape = np.array(data.get("Face"))
	euler_angle = np.array(data.get("Pose"))
	reprojectdst = np.array(data.get("b"))
	
	
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
	#reprojectdst, euler_angle = get_head_pose(shape)
	# for start, end in line_pairs:
		# cv2.line(image, reprojectdst[start], reprojectdst[end], (0, 0, 255))

	cv2.putText(image, "X: " + "{:7.2f}".format(euler_angle[0, 0]), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (255, 0, 0), thickness=2)
	cv2.putText(image, "Y: " + "{:7.2f}".format(euler_angle[1, 0]), (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (255, 0, 0), thickness=2)
	cv2.putText(image, "Z: " + "{:7.2f}".format(euler_angle[2, 0]), (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
		0.75, (255, 0, 0), thickness=2)
	cv2.imshow("line",image)
	
	key = cv2.waitKey(1) & 0xFF
  
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
# do a bit of cleanup
cv2.destroyAllWindows()
