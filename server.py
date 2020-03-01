import socket
import numpy as np
import time
import socketserver
from imageai.Detection import ObjectDetection
import os
import pickle
import cv2

global detector


class Handler(socketserver.BaseRequestHandler):
	def handle(self):
		# self.request is the TCP socket connected to the client
		imgW = 384
		imgH = 216
		self.data = self.request.recv(int(imgW*imgH*3)).strip()

		imgdata = np.fromstring(self.data,dtype=np.uint8)
		image_array = np.reshape(imgdata,(imgH,imgW,3))
 
		detected_image_array, detections = detector.detectObjectsFromImage(input_type="array", input_image=image_array , output_type="array")
		t1 = time.time()

		# detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image/test.jpg"), output_image_path=os.path.join(execution_path , "output/detection2newwhy.png"), minimum_percentage_probability=30)
		
		# print(detections)
		d = pickle.dumps(detections)

		print("{} wrote:".format(self.client_address[0]))

		# cv2.imshow("color",image_array)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()


		self.request.sendall(d)


class Server():
	def __init__(self, host: str, port: int):
		self.host = host
		self.port = port

	def start(self):
		print(self.host)
		print(self.port)
		server = socketserver.TCPServer((self.host, self.port), Handler)
		server.timeout = 1
		server.serve_forever()


if __name__ == "__main__":
	HOST = "localhost"
	PORT = 3000

	execution_path = os.getcwd()
	detector = ObjectDetection()
	detector.setModelTypeAsYOLOv3()
	detector.setModelPath( os.path.join(execution_path , "model/yolo.h5"))
	detector.loadModel()

	test_img = cv2.imread('./image/test.jpg')
	# print(test_img)
	detected_image_array, detections = detector.detectObjectsFromImage(input_type="array", input_image=test_img , output_type="array")

	print(detections)
	print('★Prepared Detector!!!!====================')
	print('★start Server!!!!====================')

	server = Server(HOST, PORT)
	server.start()