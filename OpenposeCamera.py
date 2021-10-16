import pyopenpose as op
import cv2
import argparse
from typing import List
import numpy as np

def main():
    openposeWebcam = OpenposeWebcam('http://192.168.31.157/stream.jpg')
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

        

    def capture_image(self):
        _, img = self.cap.read()
        self.frame_id += 1
        # Process Image
        self.datum.cvInputData = img
        self.opWrapper.emplaceAndPop([self.datum])

    def process_keypoints(self):
        network_output = self.datum.poseKeypoints
        if type(network_output) == np.array:
            network_output = network_output[0]
            output_array = []
            for i in range(len(network_output) - 1):
                output_array.append(network_output[i][0])
                output_array.append(network_output[i][1])
            self.output.append(output_array)


    def run(self):
        while(cv2.waitKey(1) != ord('q')):
            self.capture_image()
            self.process_keypoints()
            cv2.imshow("OpenPose", self.datum.cvOutputData)
        # output = np.array(self.output)
        # np.save('output.npy', output)        
    

if __name__=='__main__':
    main()