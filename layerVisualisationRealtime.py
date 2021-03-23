import torch
from model import ModelVis
import cv2
import torchvision.transforms as transforms
from os import environ

cap = cv2.VideoCapture(environ.get("CAMERA_2_ADDR"))

CLASSES = {0:"Nothing",1:"Something"}


# Image cropping settings
X = 700
Y = 420
SIZE = 200

# INITIALIZE THE NN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModelVis()
model.load_state_dict(torch.load("3Epochs00015lrv3_state_dict.pth",map_location=device))
model.to(device)
model.eval()



print("[ READY ] Loading done ")

while True:
    #Capture the image
    ret, frame = cap.read()
    # Cropp the image
    frame = frame[Y:Y + SIZE, X:X + SIZE]

    # Torch transforms
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Grayscale(),
        transforms.Resize((100, 100))
    ])

    #Apply the transforms
    image_data = frame
    image_data = trans(image_data)
    image_data = torch.unsqueeze(image_data,0).float()
    (images,pred) = model(image_data.to(device))

    # check if the results are over 0.5 -> class 1 else-> class 2
    result = (pred >= 0.5).item()
    print(CLASSES[result])

    # Draw the class name over the image
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(frame, CLASSES[result], (0 , 25), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    # Resize
    images = cv2.resize(images,(images.shape[1]*2,images.shape[0]*2))

    #Display the image
    cv2.imshow('Camera', images)
    #cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()
