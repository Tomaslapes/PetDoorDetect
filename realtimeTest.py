import cv2
import torch
import torchvision.transforms as transforms
from os import environ

"""
Test the classification in realtime.
"""

cap = cv2.VideoCapture(environ.get("CAMERA_2_ADDR"))

CLASSES = {0:"Nothing",1:"Something"}

# Image cropping settings
X = 700
Y = 420
SIZE = 200

# INITIALIZE THE NN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = torch.load("newData.pth",map_location=device)
model.eval()


print("[ READY ] Loading done ")

while True:
    ret, frame = cap.read()

    # Cropp the image
    frame = frame[Y:Y + SIZE, X:X + SIZE]

    # Torch transforms
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((100, 100))
    ])

    image_data = frame
    image_data = trans(image_data)
    image_data = torch.unsqueeze(image_data,0).float()
    pred = model(image_data.to(device))


    result = (pred >= 0.5).item()
    print(CLASSES[result])

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, CLASSES[result], (0 , 25), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()