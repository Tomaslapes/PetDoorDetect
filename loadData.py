from torch.utils.data import Dataset
import torch
from os import listdir
import pandas as pd
from tqdm import tqdm
import cv2

class MyDataset(Dataset):
    def __init__(self,categoriesPaths,transform = None,path = "Data/"):
        if categoriesPaths == {}:
            raise ValueError
        self.data = pd.DataFrame(columns=["Data","Label"])
        self.transform = transform

        for categ in tqdm(categoriesPaths):
            print(f"[LOADING] Category:{categ}")
            _data = []
            for categ_path in categoriesPaths[categ]:
                print(f"[LOADING] Data from :{path+categ_path}")
                lst = listdir(path+categ_path)
                lst = [path+categ_path+"/"+i for i in lst]
                _data.extend(lst)
            _labels = [categ for i in range(len(_data))]
            self.data = self.data.append(pd.DataFrame({"Data":_data,"Label":_labels},columns=["Data","Label"]),ignore_index=True)

        print("Data preparation DONE")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        image_info = self.data.iloc[index]
        image = cv2.imread(image_info["Data"])
        label = torch.tensor([image_info["Label"]])
        if self.transform:
            image = self.transform(image)

        return (image,label)



if __name__ == "__main__":
    dataLoader = MyDataset({0:["Nothing"],1:["Cat","Dog","Both"]})
    print(dataLoader.__getitem__(10))