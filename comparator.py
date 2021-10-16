
import os
import pyopenpose as op
import cv2
from itertools import zip_longest
import copy
import math

index_to_compare = [(5,2),(5,6),(2,3),(6,7),(3,4),(12,9),(5,12),(2,9),(12,13),(9,10),(14,15),(10,11)]


def main():
    # Setting the parameters to the default setttings
    params = dict()
    params['model_folder'] = "/openpose/models"

    #Start the OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    #Process Image Template:
    datum = op.Datum()
    imageToProcess = cv2.imread('/openpose/examples/media/COCO_val2014_000000000192.jpg')
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])
    output = datum.poseKeypoints 

    def grouper(iterable, n, fillvalue=None):
        "Collect data into fixed-length chunks or blocks"
        args = [iter(iterable)] * n
        return list(zip_longest(*args, fillvalue=fillvalue))

    BODY_PARTS = op.getPoseBodyPartMapping(op.BODY_25)
    POSE_PAIRS = op.getPosePartPairs(op.BODY_25)
    POSE_PAIRS = grouper(POSE_PAIRS, 2, None)

    # Unwrapping Person Keypoints from Openpose Network Output
    keypoints = []
    index_names = copy.deepcopy(BODY_PARTS)
    index_names.pop(25)
    for human in output:
        output_sub_arr = {}
        for i in range(len(human)):
            output_sub_arr[BODY_PARTS[i]]= [int(human[i][0]),int(human[i][1])]
        keypoints.append(output_sub_arr)

    # Change into coordinate list
    poses_list = []
    for pose in keypoints:
        pose = list(pose.values())
        pose = [(None,None) if x==[0,0] else tuple(x) for x in pose]
        poses_list.append(pose)

    print(poses_list)

  # Matching keypoints indices in the output of PoseNet
  # 0. Left shoulder to Right shoulder (5-2)
  # 1. Left shoulder to left elbow (5-6)
  # 2. Right shoulder to right elbow (2-3)
  # 3. Left elbow to left wrist (6-7)
  # 4. Right elbow to right wrist (3-4)
  # 5. Left hip to right hip (12-9)
  # 6. Left shoulder to left hip (5-12)
  # 7. Right shoulder to right hip (2-9)
  # 8. Left hip to left knee (12-13)
  # 9. Right hip to right knee (9-10)
  # 10. Left knee to left ankle (14-15)
  # 11.  Right knee to right ankle (10-11)
    
    
    template_values = []
    for part in index_to_compare:
        template_values.append(angle_length(poses_list[part[0]], poses_list[part[1]]))

    target_values = []
    for part in index_to_compare:
        target_values.append(angle_length(poses_list[part[0]], poses_list[part[1]]))
    
    print(template_values)
    print(target_values)



def angle_length(p1, p2):

  '''
  Input:
    p1 - coordinates of point 1. List
    p2 - coordinates of point 2. List
  Output:
    Tuple containing the angle value between the line formed by two input points 
    and the x-axis as the first element and the length of this line as the second
    element
  '''

  angle = math.atan2(- int(p2[0]) + int(p1[0]), int(p2[1]) - int(p1[1])) * 180.0 / np.pi
  length = math.hypot(int(p2[1]) - int(p1[1]), - int(p2[0]) + int(p1[0]))
  
  return round(angle), round(length)

def matching(template_kp, target_kp, angle_deviation=10, size_deviation=2):

  '''Input:
      1. template_kp - list of tuples (for the template image) containng angles 
      between particular body parts and x-axis as first elements and its sizes 
      (distances between corresponding points as second elements)
      2. target_kp - same for the target image
      3. angle_deviation - acceptable angle difference between corresponding 
      body parts in the images
      4. size_deviation - acceptable proportions difference between the images
    Output:
      List of body parts which are deviated
  '''

  devs = []

  # set an anchor size for proportions calculations - distance between shoulders
  templ_anchor = template_kp[0][1]
  targ_anchor = target_kp[0][1]

  # for each body part that we calculated angle and size for
  for i in range(len(template_kp)):

    angles = (template_kp[i][0], target_kp[i][0])
    diff_angle = max(angles) - min(angles)

    templ_size = (template_kp[i][1],templ_anchor)
    templ_size = abs(min(templ_size) / max(templ_size))

    tar_size = (target_kp[i][1], targ_anchor)
    tar_size = abs(min(tar_size) / max(tar_size))

    if diff_angle > angle_deviation:
      devs.append(i)
      print("{0} has different angle".format(i))

    elif max(tar_size,templ_size) - min(tar_size,templ_size) > size_deviation:
      devs.append(i)
      print("{0} has different size".format(i))

  return devs

if __name__ == "__main__":
    main()