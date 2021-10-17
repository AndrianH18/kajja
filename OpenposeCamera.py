import pyopenpose as op
import cv2
import argparse
from typing import List
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.utils import to_categorical

def main():
    openposeWebcam = OpenposeWebcam("http://192.168.31.214/stream.jpg")
    openposeWebcam.run()
    

class OpenposeWebcam:
    BODY_PARTS = op.getPoseBodyPartMapping(op.BODY_25)
    def __init__(self, src = 0):
            # Flags
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        self.params = dict()
        self.params['model_folder'] = "/openpose/models"
        self.params['number_people_max'] = 1

        # Add others in path?
        for i in range(0, len(args[1])):
            curr_item = args[1][i]
            if i != len(args[1])-1: next_item = args[1][i+1]
            else: next_item = "1"
            if "--" in curr_item and "--" in next_item:
                key = curr_item.replace('-','')
                if key not in self.params:  self.params[key] = "1"
            elif "--" in curr_item and "--" not in next_item:
                key = curr_item.replace('-','')
                if key not in self.params: self.params[key] = next_item
        
        #Start the OpenPose
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()

        self.cap = cv2.VideoCapture(src)
        self.datum = op.Datum()

        # Set VideoCaptureProperties 
        self.cap.set(3, 1280)    # width = 1280
        self.cap.set(4, 720)     # height = 720
        self.CAMERA_RESOLUTION_WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.CAMERA_RESOLUTION_HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.CENTER_X = self.CAMERA_RESOLUTION_WIDTH//2
        self.CENTER_Y = self.CAMERA_RESOLUTION_HEIGHT//2
        self.frame_id = 0
        self.output = []
        self.output_array = []
        self.exercise = 'squat'
        self.counter = 0
        self.flag_max = False
        self.flag_min = False 
        self.angle_dev = 7 #in degrees
        self.model =  load_model('detector/models/best.pb')
        self.prediction = 'squat'
        

    def capture_image(self):
        _, img = self.cap.read()
        self.frame_id += 1
        # Process Image
        self.datum.cvInputData = img
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))

    def process_keypoints(self):
        network_output = self.datum.poseKeypoints
        self.image = self.datum.cvOutputData
        if network_output is not None:
            if network_output.ndim:
                network_output = network_output[0]
                self.output_array = []
                for i in range(len(network_output)):
                    self.output_array.append(np.array((network_output[i][0], network_output[i][1])))
                self.output.append(self.output_array)
            else:
                self.image = cv2.putText(self.image, "No Pose detected", (int(0.75 *self.CAMERA_RESOLUTION_WIDTH), int(self.CAMERA_RESOLUTION_HEIGHT * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        


    def angle_list(self):
        # Matching keypoints indices in the output of PoseNet
        
        # 0. Angle between right shoulder, left shoulder, and left elbow (2,5,6)
        # 1. Angle between left shoulder, right shoulder, and right elbow (5,2,3)
        # 2. Angle between left shoulder, elbow, and wrist (5,6,7)
        # 3. Angle between right shoulder, elbow and wrist (2,3,4)
        # 4. Angle between right shoulder, left shoulder, and left hip (2,5,12)
        # 5. Angle between left shoudler, right shoulder, and right hip (5,2,9)
        # 6. Angle between left shoulder, left hip, right hip (5,12,9)
        # 7. Angle between right shoulder, right hip, left hip (2,9,12)
        # 8. Angle between left shoulder, hip, knee (5,12,13)
        # 9. Angle between right shoulder, hip, knee (2,9,10)
        # 10. Angle between left hip, knee, angkle (12,13,14)
        # 11. Angle between right hip, knee, angkle (9,10,11)

        self.angle_values = []
        index_to_compare = [(2,4,6), (5,2,3), (5,6,7), (2,3,4), (2,5,12), (5,2,9), (5,12,9), (2,9,12), (5,12,13),(2,9,10), (12,13,14), (9,10,11)]
        for indexes in index_to_compare:
            if len(self.output_array) != 0:
                p1 = indexes[0]
                pivot = indexes[1]
                p2 = indexes[2]

                angle = np.arccos(np.dot(self.output_array[p1]-self.output_array[pivot],self.output_array[p2]-self.output_array[pivot]) / (np.linalg.norm(self.output_array[pivot]-self.output_array[p1]) * np.linalg.norm(self.output_array[pivot]-self.output_array[p2])))
                angle = np.degrees(angle)
                self.angle_values.append(angle)
        return self.angle_values
    

    def rep_counter(self):
        if len(self.angle_values) == 0:
            return self.counter
        
        self.attention = ""
        self.image = cv2.putText(self.image, f"Exercise detected: {self.exercise}", (int(self.CAMERA_RESOLUTION_WIDTH * 0.01), int(self.CAMERA_RESOLUTION_HEIGHT * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        if self.exercise == 'dumbell-curl':
            max_angle = 170
            min_angle = 45
            # Check if the movement is correct
            if self.flag_max == True and self.flag_min== False:
                self.attention = "Curl more!"
                # print("Curl more!")
            if abs(self.angle_values[3]-max_angle)<= self.angle_dev or abs(self.angle_values[2] -max_angle)<= self.angle_dev:
                self.flag_max = True
                self.flag_min = False
            if abs(self.angle_values[3]-min_angle) <= self.angle_dev or abs(self.angle_values[2]-min_angle) <= self.angle_dev:
                self.flag_min = True

            # Set the counter
            if self.flag_max and self.flag_min == True:
                self.counter +=1
                self.flag_max = False
                self.flag_min = False
        
        elif self.exercise == 'push-up':
            max_angle = 170
            min_angle = 77

             # Set the counter
            if self.flag_max == True and self.flag_min == True:
                self.attention = "Bend more"
                self.counter += 1
                self.flag_max = False
                self.flag_min = False

            # # Check if the hip is straight
            # if abs(self.angle_values[6] - hip_angle) <= self.angle_dev or abs(self.angle_values[7]-hip_angle) <= self.angle_dev:
            #     self.flag_hip = True
            # if abs(self.angle_values[6] - hip_angle) > self.angle_dev or abs(self.angle_values[7]-hip_angle) > self.angle_dev:
            #     self.flag_hip = False
            #     print("Straighten your hip")

            # Check if your movement is correct
            if abs(self.angle_values[3]-max_angle)<= self.angle_dev or abs(self.angle_values[2] -max_angle)<= self.angle_dev:
                self.flag_max = True
                self.flag_min = False
            if abs(self.angle_values[3]-min_angle) <= self.angle_dev or abs(self.angle_values[2]-min_angle) <= self.angle_dev:
                self.flag_min = True

        elif self.exercise == 'squat':
            max_angle = 170
            min_angle =  77
            # Check if the movement is correct
            if self.flag_max == True and self.flag_min== False:
                self.attention = "Go Lower!"
                # print("Go Lower!")
            if abs(self.angle_values[10]-max_angle)<= self.angle_dev or abs(self.angle_values[11] -max_angle)<= self.angle_dev:
                self.flag_max = True
                self.flag_min = False
            if abs(self.angle_values[10]-min_angle) <= self.angle_dev or abs(self.angle_values[11]-min_angle) <= self.angle_dev:
                self.flag_min = True

            # Set the counter
            if self.flag_max and self.flag_min == True:
                self.counter +=1
                self.flag_max = False
                self.flag_min = False
        
        self.image = cv2.putText(self.image, self.attention, (int(self.CAMERA_RESOLUTION_WIDTH * 0.01), int(self.CAMERA_RESOLUTION_HEIGHT * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        self.image = cv2.putText(self.image, f"count: {self.counter}", (int(self.CAMERA_RESOLUTION_WIDTH * 0.01), int(self.CAMERA_RESOLUTION_HEIGHT * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        return self.counter
        


    def detect_exercise(self):
        print(len(self.output))
        if len(self.output)>=10:
            keypoints = [ np.array(output).flatten() for output in self.output[-10:] ]
            self.output = self.output[-10:]
            self.prediction = self.model.predict( np.array([keypoints]) )
            self.image = cv2.putText(self.image, f"Prediction: {self.prediction}", (int(self.CAMERA_RESOLUTION_WIDTH * 0.01), int(self.CAMERA_RESOLUTION_HEIGHT * 0.2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

        return

    def run(self):
        while(cv2.waitKey(1) != ord('q')):
            self.capture_image()
            self.process_keypoints()
            self.detect_exercise()
            self.angle_list()
            self.rep_counter()
            cv2.imshow("Output OpenPose", self.image)
            # cv2.imshow("OpenPose", self.datum.cvOutputData)
        # output = np.array(self.output)
        # np.save('output.npy', output)        
    

if __name__=='__main__':
    main()
    
    