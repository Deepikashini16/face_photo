from flask import Flask, render_template, url_for, request
import requests
import re
import face_recognition as fr
import os
import face_recognition
import numpy as np
from time import sleep
import cv2
import base64
from base64 import decodestring

app = Flask(__name__)

@app.route('/', methods=['POST'])
def home():
    img_string = request.form['username']
    imgdata = base64.b64decode(img_string)
    filename = 'some_image.jpg'  # I assume you have a way of picking unique filenames
    with open(filename, 'wb') as f:
        f.write(imgdata)
    names = classify_face("some_image.jpg","")
    if names == "":
        return "Unknown"
    return names


def get_encoded_faces():
    encoded = {}

    for dirpath, dnames, fnames in os.walk("./faces"):
        for f in fnames:
            if f.endswith(".jpg") or f.endswith(".png"):
                face = fr.load_image_file("faces/" + f)

                encoding = fr.face_encodings(face)[0]
                encoded[f.split(".")[0]] = encoding
    return encoded


def unknown_image_encoded(img):
    face = fr.load_image_file("faces/" + img)
    encoding = fr.face_encodings(face)[0]
    return encoding


def classify_face(im,String_names):
    String_address = ""
    faces = get_encoded_faces()
    faces_encoded = list(faces.values())
    known_face_names = list(faces.keys())

    img = cv2.imread(im, 1)

    face_locations = face_recognition.face_locations(img)
    unknown_face_encodings = face_recognition.face_encodings(img, face_locations)

    face_names = []
    for face_encoding in unknown_face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(faces_encoded, face_encoding)

        for i in range(len(matches)):
            if matches[i]:
                file_name="faces/"+known_face_names[i]+".jpg"
                with open(file_name, "rb") as img_file:
                    my_string = base64.b64encode(img_file.read())
                String_address+=str(my_string)+"#"
    return String_address

if __name__ == '__main__':
    app.run(debug=True)
