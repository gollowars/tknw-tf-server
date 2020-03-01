from imageai.Detection import ObjectDetection
import os
import cv2
import numpy as np
import pickle



imgW = 384
imgH = 216

image_array = cv2.imread('image/test.jpg')
tmp_image = image_array.tostring()
convert_array = tmp_image
imgdata = np.fromstring(convert_array,dtype=np.uint8)
convert_image_array = np.reshape(imgdata,(imgH,imgW,3))

print(type(image_array))
print(type(convert_image_array))
print(type(image_array[0][0][0]))
print(type(convert_image_array[0][0][0]))

print(image_array.dtype)
# print(imgdata.size)


# cv2.imshow("color",convert_image_array)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# execution_path = os.getcwd()
# detector = ObjectDetection()
# detector.setModelTypeAsYOLOv3()
# detector.setModelPath( os.path.join(execution_path , "model/yolo.h5"))
# detector.loadModel()

# print('★Prepared Detector!!!!====================')

# detected_image_array, detections = detector.detectObjectsFromImage(input_type="array", input_image=convert_image_array , output_type="array")


# print(detections)
# print(type(detections))

# print(detections[0]['box_points'])
# print(type(detections[0]['box_points'][0]))

# d = pickle.dumps(detections)

# detections = pickle.loads(d)

# print('load = ================')
# print(detections)

# print(type(detections[0]['box_points'][0]))
