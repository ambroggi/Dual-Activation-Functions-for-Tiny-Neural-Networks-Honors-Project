# Created by: Alexandre Broggi 2023

import pandas as pd
import DataSource
import ModelStruct
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision


def main():
    batchNorm = False
    beans = DataSource.beansData()
    train, test = random_split(beans, [len(beans) - int(len(beans) * 0.1), int(len(beans) * 0.1)])
    beans_loader = DataLoader(train, batch_size=1000, shuffle=True)
    modules = [ModelStruct.agglomerativeModel(beans.numFeatures, beans.numClasses, x, 50, 0.1, norm=batchNorm, layerSize=10, numLayers=2) for x in ModelStruct.activationLists]

    dataframe = pd.DataFrame([["Beans", 0, 0, 0, 0]], columns=["Name", "F1", "Precision", "Recall", "BatchNorm"])

    for x in tqdm(range(50)):
        for module in modules:
            module.trainCycle(beans_loader)

    beans_loader = DataLoader(test, batch_size=100000, shuffle=True)

    for module in modules:
        module.eval()
        scores = (0, 0, 0)
        for bean in beans_loader:
            # for x in module.modList:
            #     x.debug_confMat(bean)
            scores = module(bean)
        dataframe = pd.concat([dataframe, pd.DataFrame([[module.activation, scores[0], scores[1], scores[2], batchNorm, module.count, module.layerSize, module.numLayers]], columns=["Name", "F1", "Precision", "Recall", "BatchNorm", "Count", "LayerSize", "NumLayers"])])

    dataframe.to_csv("FinalValues.csv", mode="a")


def torchvisionversion():
    batchNorm = False
    train = torchvision.datasets.MNIST("MNIST", train=True, download=True, transform=torchvision.transforms.ToTensor())
    test = torchvision.datasets.MNIST("MNIST", train=False, download=True, transform=torchvision.transforms.ToTensor())
    loader = DataLoader(train, batch_size=1000, shuffle=True)
    modules = [ModelStruct.agglomerativeModel(784, 10, x, 50, 0.1, norm=batchNorm, layerSize=10, numLayers=2) for x in ModelStruct.activationLists]

    dataframe = pd.DataFrame([["MNIST", 0, 0, 0, 0]], columns=["Name", "F1", "Precision", "Recall", "BatchNorm"])

    for x in tqdm(range(50)):
        for module in modules:
            module.trainCycle(loader)

    loader = DataLoader(test, batch_size=100000, shuffle=True)

    for module in modules:
        module.eval()
        scores = (0, 0, 0)
        for bean in loader:
            # for x in module.modList:
            #     x.debug_confMat(bean)
            scores = module(bean)
        dataframe = pd.concat([dataframe, pd.DataFrame([[module.activation, scores[0], scores[1], scores[2], batchNorm, module.count, module.layerSize, module.numLayers]], columns=["Name", "F1", "Precision", "Recall", "BatchNorm", "Count", "LayerSize", "NumLayers"])])

    dataframe.to_csv("FinalValues.csv", mode="a")


if __name__ == "__main__":
    torchvisionversion()
