import torch
from model import ModelVis
import cv2
import torchvision.transforms as transforms

CLASSES = {0:"Nothing",1:"Something"}

IMAGE = "WeZmyWFd95"


# Image cropping settings
X = 700
Y = 420
SIZE = 200

# INITIALIZE THE NN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ModelVis()
#3Epochs00015lrv3
model.load_state_dict(torch.load("newData_state_dict.pth",map_location=device))
model.to(device)
model.eval()



print("[ READY ] Loading done ")

frame = cv2.imread(f"RealtimeClassifDataCol/Something/{IMAGE}.jpg")

    # Cropp the image
#frame = frame[Y:Y + SIZE, X:X + SIZE]

    # Torch transforms
trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((100, 100))
])

image_data = frame
image_data = trans(image_data)
image_data = torch.unsqueeze(image_data,0).float()
(images,pred) = model(image_data.to(device))



result = (pred >= 0.5).item()
print(CLASSES[result])

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(frame, CLASSES[result], (0 , 25), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
images = cv2.resize(images,(images.shape[1]*2,images.shape[0]*2))
cv2.imshow('Camera', images)
cv2.waitKey(0)
cv2.imshow('Camera', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()


if input("Save image. Y/n") == "Y":
    name = input("name")
    images = images*255
    cv2.imwrite(f"Graphs/{IMAGE}{name}.png",images)
