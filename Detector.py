from csv import reader
import yagmail
from os import environ
from time import sleep
from time import time
import cv2
import torch
import torchvision.transforms as transforms
import requests
import json


# GLOBAL VARS
CHECK_PERIOD = 7 # checks for a pet every X seconds
MAIL_SEND_TIMEOUT = 30 # waits for X seconds after sending out all the emails
INCLUDE_IMAGE = True
REPEAT_DETECT = 2
MAX_SEND_ROW = 4 # How many times to send the notification before long timeout
LONG_TIMEOUT = 180

CROP_X = 700
CROP_Y = 420
CROP_SIZE = 200

# Set up for notifications
serverToken = environ.get("FIREBASE_PET_TOKEN")
deviceToken = '/topics/all'

headers = {
        'Content-Type': 'application/json',
        'Authorization': 'key=' + serverToken,
      }

body = {
          'notification': {'title': 'Pes / Kočka je u dveří',
                            'body': 'Detekuji kočku nebo psa'},
          'to':deviceToken,
          'priority': 'high',
        #   'data': dataPayLoad,
        }

# Set up for emails
yagmail.register(environ.get("DOG_MAIL"), environ.get("DOG_MAIL_PASS"))

SENDER_EMAIL = environ.get("DOG_MAIL")

RECIEVER_EMAIL = []
with open("emails.csv") as f:
    _emails = reader(f)
    for row in _emails:
        RECIEVER_EMAIL.append(row)


BODY = """
Pes nebo kocka je u dveri!!
"""

# Create a device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the model
model = torch.load("2Epochs0005lrv2.pth",map_location=device)
model.eval()


def detect(model):
    cap = cv2.VideoCapture(environ.get("CAMERA_2_ADDR"))
    _, frame = cap.read()

    # Crop the image
    frame = frame[CROP_Y:CROP_Y + CROP_SIZE, CROP_X:CROP_X + CROP_SIZE]

    # Torch transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((100, 100))
    ])

    image_data = frame
    image_data = transform(image_data)
    image_data = torch.unsqueeze(image_data, 0).float()
    pred = model(image_data.to(device))
    pred = (pred >= 0.5).item()
    print("[ LOG ] Pet present: ",pred)
    cap.release()
    return pred

print("[ READY ] Loading has finished!")

def send_notification(body):
    response = requests.post("https://fcm.googleapis.com/fcm/send",headers = headers, data=json.dumps(body))
    print(f"[ SEND ] notification sent. Got back: {response.status_code}")

def send_email(sender_email,reciever_email,body):
    print("[ SEND ] sending email")
    yag = yagmail.SMTP(sender_email)
    for person in reciever_email:
        yag.send(
            to=person,
            subject="Pes nebo kocka cekaji!",
            contents=body,
            #attachments="Data/Dog/cat915.jpg",
        )
    print("[ SEND ] done")

RUNNING = True
LAST_SENT_TIME = None
NUM_SENT = 0
while RUNNING:
    try:
        prediction = detect(model)

        if  prediction == 1:
            if LAST_SENT_TIME is None:
                _ok = True
                # Repeat for n times to make sure that it wasnt a false positive
                for repeat in range(REPEAT_DETECT):
                    pred = detect(model)
                    if pred == 1:
                        continue
                    else:
                        _ok = False
                if _ok:
                    LAST_SENT_TIME = time()
                    send_notification(body)
                    send_email(SENDER_EMAIL,RECIEVER_EMAIL,BODY)
                    NUM_SENT += 1
            else:
                now = time()
                if (now - LAST_SENT_TIME) >= MAIL_SEND_TIMEOUT:
                    if NUM_SENT < MAX_SEND_ROW:
                        send_notification(body)
                        send_email(SENDER_EMAIL,RECIEVER_EMAIL,BODY)
                        LAST_SENT_TIME = None
                        NUM_SENT += 1
                    elif (now - LAST_SENT_TIME) >= LONG_TIMEOUT:
                        NUM_SENT = 0
        else:
            if LAST_SENT_TIME is not None:
                LAST_SENT_TIME = None
            NUM_SENT = 0

        sleep(CHECK_PERIOD)


    except Exception as e:
        print(e)
        RUNNING = False
    except KeyboardInterrupt:
        RUNNING = False
        print("Shutting down!")