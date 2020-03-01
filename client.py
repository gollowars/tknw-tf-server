import socket
import time
import cv2
import numpy as np
import pickle

class Client():
    def __init__(self):
        print('client')
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def connect(self, host:str , port: int):
        self.client.connect((host, port))
    
    def send(self, msg: str):
        self.client.send(msg.encode('utf-8'))
        response = self.client.recv(4096)
        return response
    
    def sendDetectImg(self, img: np.ndarray):
        dist = img.tostring()
        self.client.send(dist)
        response = self.client.recv(4096)
        detections = pickle.loads(response)

        return detections




if __name__ == "__main__":
    host = "localhost"
    port = 3000


    image_array = cv2.imread('image/test.jpg')

    t1 = time.time()
    print(image_array.size)

    client = Client()
    client.connect(host, port)
    detections = client.sendDetectImg(image_array)
    t2 = time.time()
    elapsed = t2-t1
    print("time :", elapsed)
    print(detections)

