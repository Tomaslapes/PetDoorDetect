import cv2
from os import listdir
from os import environ

'''
Just a quick tool to make data collection easier.
'''

cap = cv2.VideoCapture(environ.get("CAMERA_2_ADDR"))

CLASSES = {"cat":("Data/Cat/",0),"dog":("Data/Dog/",0),"nothing":("Data/Nothing/",0)}

#Image cropping settings
X = 700
Y = 420
SIZE = 200

for my_class in CLASSES:
    #files = listdir(CLASSES[my_class][0])
    #files.sort()
    #print(files)
    #print(my_class, sorted( listdir(CLASSES[my_class][0]) )[-1].strip(my_class).strip(".jpg"))
    #file_number = int(listdir(CLASSES[my_class][0])[-1].strip(my_class).strip(".jpg"))
    file_number = len(listdir(CLASSES[my_class][0]))
    CLASSES[my_class] = (CLASSES[my_class][0],file_number+250)

print(CLASSES)
print("[ READY ] press D to capture a DOG, C to capture a CAT, and X to capture NOTHING. Exit with Q ")

while True:
    ret, frame = cap.read()

    # Cropp the image
    frame = frame[Y:Y+SIZE,X:X+SIZE]

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    # TODO put this into a for loop
    if cv2.waitKey(1) & 0xFF == ord('d'):
        cv2.imwrite(f"{CLASSES['dog'][0]}dog{CLASSES['dog'][1]}.jpg",frame)

        CLASSES['dog'] = (CLASSES['dog'][0],CLASSES['dog'][1]+1)
        print("DOG image saved!")

    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite(f"{CLASSES['cat'][0]}cat{CLASSES['cat'][1]}.jpg",frame)
        
        CLASSES['cat'] = (CLASSES['cat'][0],CLASSES['cat'][1]+1)
        print("CAT image saved!")

    if cv2.waitKey(1) & 0xFF == ord('x'):
        cv2.imwrite(f"{CLASSES['nothing'][0]}nothing{CLASSES['nothing'][1]}.jpg",frame)

        CLASSES['nothing'] = (CLASSES['nothing'][0],CLASSES['nothing'][1]+1)
        print("NOTHING image saved!")

cv2.destroyAllWindows()