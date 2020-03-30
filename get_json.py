import sys
import os
sys.path.append("/home/natalia/openpose/build/examples/tutorial_api_python")
from my_pose_attempt_1 import create_pose_json

for num in range(1, 3):
    path = "/home/natalia/openpose/examples/media/Turn_on/Turn_on{}".format(num)
    train_path = "/home/natalia/openpose/examples/media/Turn_on/Train_data"
    try:
        os.mkdir(train_path + '/' "Turn_on{}".format(num))
    except:
        pass
    images = os.listdir(path)
    for img in images:
        dir =  train_path + '/' + "Turn_on{}/".format(num)
        file = "Turn_on{}_".format(num) + img[:-4] + ".json"
        create_pose_json(path + '/' + img, dir, file)