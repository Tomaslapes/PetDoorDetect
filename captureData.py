import cv2
import torch
import torchvision.transforms as transforms
from time import sleep
from django.utils.crypto import get_random_string
from os import environ

'''
Goal of this program is to capture data in 1 minute intervals
and try to classify the image. This will later server to identify
scenarios that are more difficult for the network
'''
#Image cropping settings
X = 700
Y = 420
SIZE = 200

# INITIALIZE THE NN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = torch.load("newData.pth",map_location=device)
model.eval()

CLASSES = {0:"Nothing",1:"Something"}

print("[ READY ] Loading done ")

CAMERA_ADDR = environ.get('CAMERA_2_ADDR')

while True:
    OK = False

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((100, 100))
    ])

    while not OK:
        try:
            cap = cv2.VideoCapture(CAMERA_ADDR)

            ret, frame = cap.read()

            # Crop the image
            frame = frame[Y:Y+SIZE,X:X+SIZE]
            cap.release()

            image_data = frame
            image_data = trans(image_data)
            image_data = torch.unsqueeze(image_data,0).float()
            pred = model(image_data.to(device))
            result = (pred >= 0.5).item()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            random_name = get_random_string(length=10)

            cv2.imwrite(f"RealtimeClassifDataCol/{CLASSES[result]}/{random_name}.jpg",frame)

            cap.release()
            OK =True
        except Exception as e:
            if cap is not None:
                print(f"cap value: {cap}>>","Cap released!")
                cap.release()
            print(e)

    sleep(60)

cv2.destroyAllWindows()
