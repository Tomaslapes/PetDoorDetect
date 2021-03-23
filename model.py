import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=10,kernel_size=10)
        self.conv2 = nn.Conv2d(in_channels=10,out_channels=20,kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=20,out_channels=40,kernel_size=10,stride=2)

        self.fc1 = nn.Linear(60840,1024)#40*40*40
        self.fc2 = nn.Linear(1024,128)
        self.out = nn.Linear(128,1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)
        
        x = x.reshape(-1,60840)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.out(x)
        return x


class ModelVis(nn.Module):
    ''' Model which also outputs a grid image of each convolutional layer

    '''
    def __init__(self):
        super(ModelVis, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=10, stride=2)

        self.fc1 = nn.Linear(60840, 1024)  # 40*40*40
        self.fc2 = nn.Linear(1024, 128)
        self.out = nn.Linear(128, 1)

    def create_grid(self,x):
        images = torch.zeros((int(x.shape[1] / 5 * x.shape[2]), int(x.shape[2] * 5)))

        x_vis = x[0]
        step = x.shape[2]
        y = 0
        X = 0
        for i, image in enumerate(x_vis):
            # print(f"X{X},y{y}", images)
            # print(images[y:y + step, X:X + step].shape)
            # print(image.cpu().detach().shape)
            if i % 5 == 0 and i != 0:
                y += step
                X = 0
            images[y:y + step, X:X + step] = image.cpu().detach()
            if i % 5 != 0:
                X += step
            # cv2.imshow("first conv",cv2.resize(image.cpu().detach().numpy(),(255,255)))
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        images = images.numpy()
        return images

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        images = self.create_grid(x)

        x = self.conv2(x)
        x = F.relu(x)


        x = self.conv3(x)
        x = F.relu(x)



        x = x.reshape(-1, 60840)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.out(x)
        return (images,x)
