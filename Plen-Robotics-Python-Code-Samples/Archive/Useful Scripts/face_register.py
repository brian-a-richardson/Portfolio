#!/usr/bin/python
# -*- coding: utf-8 -*-

# This script is used for registering faces from a folder of JPG images

import os
import glob
import json

import rospy
import actionlib
from global_msgs.msg import AsyncCommandAction, AsyncCommandGoal, AsyncCommandResult

IMAGE_FOLDER = "/tmp/photos"

def main():
    rospy.init_node("auto_register")
    recognize_client = actionlib.SimpleActionClient("/application/face_sdk/face_recognition/async_bulk_register_edit", AsyncCommandAction)
    data = get_data()
    recognize_client.wait_for_server()
    recognize_client.send_goal(AsyncCommandGoal(arguments=json.dumps(data)))
    recognize_client.wait_for_result()
    print recognize_client.get_result()

def get_data():
    data = []
    for filepath in glob.glob(IMAGE_FOLDER+"/*.jpg"):
        data_i = {"photoUrl":filepath}
        data_i["name"] = os.path.splitext(os.path.basename(filepath))[0]
        data.append(data_i)
    return data

main()