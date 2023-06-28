#!/usr/bin/python3.8

"""
This script is meant to detecting the patient's motion thereby alerting the staff incharge.
"""

import threading

import cv2
import pyttsx3


def voice_alarm(sound)->None:
    """Alert the user

    Args:
        sound (obj): it is a sound engine that process the text to speech task.
    """
    try:
        sound.say("patient is moving")
        sound.runAndWait()
    except RuntimeError as e:
        pass

STATUS_LIST = [None, None]

# Initiate the text to speech package; extract the desired voice and set it as
# property for our script.
alarm_sound = pyttsx3.init()    # Initiates the voice object.
voices = alarm_sound.getProperty('voices')  # Getting the available voices.
alarm_sound.setProperty('voice', voices[11].id) # Setting the desired voice
# Setting the desired  voice rate at 175 words per minute.
alarm_sound.setProperty('rate', 175)

# Initialize a video capture object
video = cv2.VideoCapture(0) # Video capturing starts in 1st cam.
INITIAL_FRAME = None    # Initial frame in the video set to None.

while True:
    _, frame = video.read()
    frame = cv2.flip(frame, 1)
    STATUS = 0

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_frame = cv2.GaussianBlur(gray_frame, (13, 13), 0)
    # cv2.imshow("gussian", blur_frame)   # test code

    if INITIAL_FRAME is None:
        INITIAL_FRAME = gray_frame
        continue

    delta_frame = cv2.absdiff(INITIAL_FRAME, blur_frame)
    threshold_frame = cv2.threshold(delta_frame, 50, 255, cv2.THRESH_BINARY)[1]

    (contours, _) = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # DO: note the point of change, if found draw a reactangle over the region
    # of changes.
    for c in contours:

        if cv2.contourArea(c) < 10000:
            continue
        STATUS = STATUS + 1
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
    STATUS_LIST.append(STATUS)  # type: ignore

    # DO: start alerting concurrently  if the STATUS_LIST get updated
    if STATUS_LIST[-1] >= 1 and STATUS_LIST[-2] == 0:  # type: ignore

        alarm = threading.Thread(target=voice_alarm, args=(alarm_sound,))
        alarm.start()
        # alarm.join()

    cv2.imshow('motion detector', frame)    # Display the video
    # DO: exit on pressing q

    if cv2.waitKey(1) == ord('q'):
        break

# DO: close the videp capturing object, destory all the GUI windows and stop
# the alert.
alarm_sound.stop()
video.release()
cv2.destroyAllWindows()
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# # EXPLAINATION OFF THIS PATIENT MOVEMENT DETECTION.
-----------------------------------------------------
# PATIENT MOVEMENT DETECTION

# Synopsis:
# This project aims to develop a detection system that is based on the single patient
# movements using the computer vision technology.
# How is it implemented?
# We have used one of the most popular programming languages ‘Python’ to develop this
# software. Since it is based on computer vision, the software utilises OpenCV which is a
# popular open source video processing library written in C++ language.
-------------------------------------------------------------------------------------------------
# Use case:
-----------

# The ultimate goal for us is:
# ● To automate the process of motion detection of the most care needed patients.
# ● Reduction of the work burden of staff in charge(nurses or staff related to care taking).
# ● Always on active motion detection for patients with high priority.
--------------------------------------------------------------------------------------------------
# Project algorithmics overview:
-------------------------------
# 1. START
# 2. GET live video footage.
# a. SET an initial frame.
# b. COMPARE upcoming frame with initial frame.
# c. CALCULATE the difference in the frames.
# d. CONVERT to binary images.
# 3. SET a threshold value for the minimal patient's
# movements.
# 4. SET contours from the binary frames.
# a. SET rectangles over the changes of contours.
# b. ALERT with sound “Patient is moving” when motion id found.
# 5. TERMINATE program when presses ‘q’ key.
# 6. STOP.
---------------------------------------------------------------------------------------------------
# Libraries used in the project:
--------------------------------
# ➢ OpenCV-python:
# ○ An open source video processing library.
# ➢ Thread:
# ○ An in-built module for creating threads in the execution of the program.
# ➢ Numpy:
# ○ A numerical calculation package written in ‘C’.
# ○ Performs various multidimensional array calculations and scientific
# calculations with extremely fast speed.
# ○ This package is required by opencv to perform the operations on the images
# from the video.

# ➢ Pyttsx3:
# ○ An offline text to speech library used for text based audio conversion.
-----------------------------------------------------------------------------------------------------