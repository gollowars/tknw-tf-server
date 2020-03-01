from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "model/yolo.h5"))
detector.loadModel()



print('★Prepared Detector!!!!====================')


detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "image/test2.jpg"), output_image_path=os.path.join(execution_path , "output/detection2.jpg"), minimum_percentage_probability=30)


for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")