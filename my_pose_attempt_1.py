# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import json

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    #parser.add_argument("--image_path", default="../../../examples/media/COCO_val2014_000000000192.jpg")
    parser.add_argument("--image_path", default="../../../examples/media/Danya.png", help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"
    params["write_json"] = "../../../results/"
    params["hand"] = True

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Process Image
    datum = op.Datum()
    imageToProcess = cv2.imread(args[0].image_path)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    poseModel = op.PoseModel.BODY_25
    print(op.getPoseBodyPartMapping(poseModel))
    #print(op.getPoseNumberBodyParts(poseModel))
    #print(op.getPosePartPairs(poseModel))
    #print(op.getPoseMapIndex(poseModel))

    poseModel = {0: 'Nose', 1: 'Neck', 2: 'RShoulder', 3: 'RElbow', 4: 'RWrist', 5: 'LShoulder', 6: 'LElbow', 7: 'LWrist', 8: 'MidHip',
                 9: 'RHip', 10: 'RKnee', 11: 'RAnkle', 12: 'LHip', 13: 'LKnee', 14: 'LAnkle', 15: 'REye', 16: 'LEye', 17: 'REar',
                 18: 'LEar', 19: 'LBigToe', 20: 'LSmallToe', 21: 'LHeel', 22: 'RBigToe', 23: 'RSmallToe', 24: 'RHeel', 25: 'Background'}
    merged = {}
    for person in range(len(datum.poseKeypoints)):
        persnum = "Person{}".format(person)
        parts = [{'bodypart': poseModel[bodypart], 'X': str(datum.poseKeypoints[person][bodypart][0]), 'Y': str(datum.poseKeypoints[person][bodypart][1]),
                'Confidence': str(datum.poseKeypoints[person][bodypart][2])} for bodypart in range(len(datum.poseKeypoints[person]))]
        lefthand = []
        for part in range(int((datum.handKeypoints[0].size)/3)):
            lefthand.append({'X' : str(datum.handKeypoints[0][part][0]), 'Y': str(datum.handKeypoints[0][part][1])})
        leftHands = ([{'LeftHandParts': {'X' : str(datum.handKeypoints[person][0][part][0]), 'Y': str(datum.handKeypoints[person][0][part][1])} for part in range(datum.handKeypoints[0].size)}])
        parts.append(leftHands)
       # parts.append({'RightHand': str(datum.handKeypoints[1])})
        merged[persnum] = parts
    res = json.dumps(merged,               sort_keys=False, indent=4, separators=(',', ': '))

    with open("data_file.json", "w") as write_file:
        json.dump(merged, write_file, sort_keys=False, indent=4, separators=(',', ': '))
    print(res)
    '''for person in range(len(datum.poseKeypoints)):
        print("Person № {}\n".format(person))
        for bodypart in range(len(poseModel)-1):
            print("{}: x = {}; y = {}, c = {}  \n".format(poseModel[bodypart], datum.poseKeypoints[person][bodypart][0], datum.poseKeypoints[person][bodypart][1],  datum.poseKeypoints[person][bodypart][2] ))
    '''
    print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    # Display Image
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.5.1 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)
except Exception as e:
    print(e)
    sys.exit(-1)
