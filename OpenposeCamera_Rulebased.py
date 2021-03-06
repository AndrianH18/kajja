import pyopenpose as op
import cv2
import argparse
from typing import List
import numpy as np

def main():
    openposeWebcam = OpenposeWebcam(0)
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
        self.exercise = 'none'
        self.counter = 0
        self.imaginary_counter = [0] * 3
        self.dumbell_flag_max = False
        self.dumbell_flag_min = False
        self.pushup_flag_max = False
        self.pushup_flag_min = False
        self.squat_flag_max = False
        self.squat_flag_min = False
        self.angle_dev = 7 #in degrees

        

    def capture_image(self):
        _, img = self.cap.read()
        self.frame_id += 1
        # Process Image
        self.datum.cvInputData = img
        self.opWrapper.emplaceAndPop([self.datum])

    def process_keypoints(self):
        network_output = self.datum.poseKeypoints
        self.image = self.datum.cvOutputData
        if network_output.ndim:
            network_output = network_output[0]
            self.output_array = []
            for i in range(len(network_output) - 1):
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
        # 10. Angle between left hip, knee, angle (12,13,14)
        # 11. Angle between right hip, knee, angle (9,10,11)
        # 12. Angle between midhip, midshoulder, left elbow (8,1,3)
        # 13. Angle between midhip, midshoulder, right elboe (8,1,6)

        self.angle_values = []
        index_to_compare = [(2,4,6), (5,2,3), (5,6,7), (2,3,4), (2,5,12), (5,2,9), (5,12,9), (2,9,12), (5,12,13),(2,9,10), (12,13,14), (9,10,11), (8,1,3),(8,1,6)]
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
        self.attention = ""
        self.image = cv2.putText(self.image, f"Exercise detected: {self.exercise}", (int(self.CAMERA_RESOLUTION_WIDTH * 0.01), int(self.CAMERA_RESOLUTION_HEIGHT * 0.05)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        # self.exercise = "dumbell-curl"
        # if self.exercise == 'dumbell-curl':
        max_angle = 170
        min_angle = 45
        # Check if the movement is correct
        if self.dumbell_flag_max == True and self.dumbell_flag_min== False:
            self.attention = "Curl more!"
            # print("Curl more!")
        if abs(self.angle_values[3]-max_angle)<= self.angle_dev or abs(self.angle_values[2] -max_angle)<= self.angle_dev:
            self.dumbell_flag_max = True
            self.dumbell_flag_min = False
        if abs(self.angle_values[3]-min_angle) <= self.angle_dev or abs(self.angle_values[2]-min_angle) <= self.angle_dev:
            self.dumbell_flag_min = True

        # Set the counter
        if self.dumbell_flag_max and self.dumbell_flag_min == True:
            # self.counter +=1
            self.dumbell_flag_max = False
            self.dumbell_flag_min = False
            self.imaginary_counter[0] += 1
            self.counter = self.imaginary_counter[0]
            self.exercise = "dumbell_curl"

        # self.exercise = "push-up"
        # if self.exercise == 'push-up':
        max_angle = 80
        min_angle = 10

            # Set the counter
        if self.pushup_flag_max == True and self.pushup_flag_min == True:
            self.attention = "Bend more"
            # self.counter += 1
            self.pushup_flag_max = False
            self.pushup_flag_min = False
            self.imaginary_counter[1] += 1
            self.counter = self.imaginary_counter[1]
            self.exercise = "push-up"

        # # Check if the hip is straight
        # if abs(self.angle_values[6] - hip_angle) <= self.angle_dev or abs(self.angle_values[7]-hip_angle) <= self.angle_dev:
        #     self.flag_hip = True
        # if abs(self.angle_values[6] - hip_angle) > self.angle_dev or abs(self.angle_values[7]-hip_angle) > self.angle_dev:
        #     self.flag_hip = False
        #     print("Straighten your hip")

        # Check if your movement is correct
        if abs(self.angle_values[12]-max_angle)<= self.angle_dev or abs(self.angle_values[13] -max_angle)<= self.angle_dev:
            self.pushup_flag_max = True
            self.pushup_flag_min = False
        if abs(self.angle_values[12]-min_angle) <= self.angle_dev or abs(self.angle_values[13]-min_angle) <= self.angle_dev:
            self.pushup_flag_min = True

        # self.exercise = "squat"
        # if self.exercise == 'squat':
        max_angle = 180
        min_angle =  77
        # Check if the movement is correct
        if self.squat_flag_max == True and self.squat_flag_min== False:
            self.attention = "Go Lower!"
            # print("Go Lower!")
        if abs(self.angle_values[10]-max_angle)<= self.angle_dev or abs(self.angle_values[11] -max_angle)<= self.angle_dev:
            self.squat_flag_max = True
            self.squat_flag_min = False
        if abs(self.angle_values[10]-min_angle) <= self.angle_dev or abs(self.angle_values[11]-min_angle) <= self.angle_dev:
            self.squat_flag_min = True

        # Set the counter
        if self.squat_flag_max and self.squat_flag_min == True:
            # self.counter +=1
            self.squat_flag_max = False
            self.squat_flag_min = False
            self.imaginary_counter[2] += 1
            self.counter = self.imaginary_counter[2]
            self.exercise = "squat"
        
        
        self.image = cv2.putText(self.image, self.attention, (int(self.CAMERA_RESOLUTION_WIDTH * 0.01), int(self.CAMERA_RESOLUTION_HEIGHT * 0.15)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        self.image = cv2.putText(self.image, f"count: {self.counter}", (int(self.CAMERA_RESOLUTION_WIDTH * 0.01), int(self.CAMERA_RESOLUTION_HEIGHT * 0.1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        return self.counter
        




    def run(self):
        while(True):
            self.capture_image()
            self.process_keypoints()
            self.angle_list()
            self.rep_counter()
            cv2.imshow("Output OpenPose", self.image)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.imaginary_counter = [0] * 3
            # cv2.imshow("OpenPose", self.datum.cvOutputData)
        # output = np.array(self.output)
        # np.save('output.npy', output)        
    

if __name__=='__main__':
    main()
    
    