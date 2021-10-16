import pyopenpose as op
import cv2
import argparse

def main():
    openposeWebcam = OpenposeWebcam()
    openposeWebcam.run()

class OpenposeWebcam:
    def __init__(self, src = 0):
            # Flags
        parser = argparse.ArgumentParser()
        args = parser.parse_known_args()

        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        self.params = dict()
        self.params['model_folder'] = "/openpose/models"

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

        

    def capture_image(self):
        _, img = self.cap.read()
        # Process Image
        self.datum.cvInputData = img
        self.opWrapper.emplaceAndPop([self.datum])

    # def get_keypoints(self):
    #     self.datum.poseKeypoints

    def run(self):
        while(cv2.waitKey(1) != ord('q')):
            self.capture_image()
            # self.get_keypoints()
            cv2.imshow("OpenPose", self.datum.cvOutputData)
    

if __name__=='__main__':
    main()