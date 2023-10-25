# Created by: Alexandre Broggi 2023

import pandas as pd
import DataSource
import ModelStruct
from torch.utils.data import DataLoader
from tqdm import tqdm


def main():
    batchNorm = False
    beans = DataSource.beansData()
    beans_loader = DataLoader(beans, batch_size=1000, shuffle=True)
    modules = [ModelStruct.agglomerativeModel(beans.numFeatures, beans.numClasses, x, 15, 0.1, norm=batchNorm) for x in ModelStruct.activationLists]

    dataframe = pd.DataFrame([["Blank", 0, 0, 0, 0]], columns=["Name", "F1", "Precision", "Recall", "BatchNorm"])

    for x in tqdm(range(50)):
        for module in modules:
            module.trainCycle(beans_loader)

    beans_loader = DataLoader(beans, batch_size=100000, shuffle=True)

    for module in modules:
        module.eval()
        scores = (0, 0, 0)
        for bean in beans_loader:
            # for x in module.modList:
            #     x.debug_confMat(bean)
            scores = module(bean)
        dataframe = pd.concat([dataframe, pd.DataFrame([[module.activation, scores[0], scores[1], scores[2], batchNorm, module.count]], columns=["Name", "F1", "Precision", "Recall", "BatchNorm"])])

    dataframe.to_csv("FinalValues.csv", mode="a")


if __name__ == "__main__":
    main()
