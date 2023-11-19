# Created by: Alexandre Broggi 2023

import requests
import zipfile
import io
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
# https://archive.ics.uci.edu/dataset/602/dry+bean+dataset

beans = {"file_adress": "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip", "File": "DryBeanDataset/Dry_Bean_Dataset.xlsx", "Command": pd.read_excel}
flowers = {"file_adress": "https://archive.ics.uci.edu/static/public/53/iris.zip", "File": "iris.data", "Command": lambda x: pd.read_csv(x, names=["sepal length", "sepal width", "petal length", "petal width", "Class"])}
ds = beans
file_adress = ds["file_adress"]
file = ds["File"]


class beansData(Dataset):
    def __init__(self):
        if not os.path.exists(file):
            zipfile.ZipFile(io.BytesIO(requests.get(file_adress, stream=True).content)).extractall()
        self.df = ds["Command"](file)
        self.classdict = {i: x for x, i in enumerate(self.df["Class"].unique())}
        self.numClasses = len(self.classdict)
        self.numFeatures = len(self.df.iloc[0]) - 1
        dictclass = {self.classdict[y]: y for y in self.classdict.keys()}
        for y in dictclass.keys():
            self.classdict[y] = dictclass[y]

        # arr = []
        # for name in self.df["Class"].unique():
        #     arr.append(self.df[self.df["Class"] == name].sample(100))

        # self.df = pd.concat(arr)
        print(self.df["Class"].value_counts())

        # print(self.df[self.df["Class"] == "SEKER"].describe())
        # print(self.df[self.df["Class"] == "BARBUNYA"].describe())
        # self.df = self.df[(self.df["Class"] == self.classdict[0]) | (self.df["Class"] == self.classdict[1])]
        # print(self.df.head())

    def __getitem__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        item = self.df.iloc[[index]]
        label = self.classdict[item.pop("Class").item()]
        # return torch.tensor(label).unsqueeze(0), torch.tensor(label), torch.tensor(index)  # TODO: REMOVE THIS IT IS JUST FOR TESTING
        return torch.tensor(item.to_numpy()[0]), torch.tensor(label), torch.tensor(index)

    # def __getitems__(self, index) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    #     items = self.df.iloc[index]
    #     labels = items.pop("Class").apply(lambda x: self.classdict[x]).to_numpy()
    #     return torch.tensor(items.to_numpy()), torch.tensor(labels), torch.tensor(index)

    def __len__(self) -> int:
        return len(self.df)


if __name__ == "__main__":

    train = torchvision.datasets.MNIST("MNIST", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test = torchvision.datasets.MNIST("MNIST", train=False, download=True, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(train, batch_size=1000, shuffle=True)

    totals_train = torch.zeros(10)
    for batch in loader:
        totals_train += batch[1].bincount()
    print(totals_train)

    loader = DataLoader(test, batch_size=1000, shuffle=True)

    totals_test = torch.zeros(10)
    for batch in loader:
        totals_test += batch[1].bincount()
    print(totals_test)

    df = ds["Command"](file)
    classdict = {i: x for x, i in enumerate(df["Class"].unique())}
    # pd.Series.apply()
    df["Class"] = df["Class"].apply(lambda x: classdict[x])
    print(df.corr())
    print(df["Class"].describe())

    beans = beansData()

    beans_loader = DataLoader(beans, batch_size=100, shuffle=True)

    for bean in beans_loader:
        print(bean)

    print(pd.get_dummies(beans.df, columns=["Class"]).corr())
