# Created by: Alexandre Broggi 2023
import torch
from torch import nn
from sklearn.metrics import (precision_score, recall_score, f1_score)
from sklearn.metrics import confusion_matrix


class MultiActivation(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modList = nn.ModuleList()
        for funct in modules:
            self.modList.append(activationDict[funct]())

    def forward(self, x):
        xs = [y(x) for y in self.modList]
        # test = torch.concat(xs, dim=1)
        return torch.concat(xs, dim=1)


class SingleActivation(nn.Module):
    def __init__(self, module):
        super().__init__()
        if isinstance(module, list):
            self.mod = module[0]()
        else:
            self.mod = module()

    def forward(self, x):
        xs = self.mod(x)
        # test = torch.concat([xs], dim=1)

        return xs


class Cosign(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)


class NoChange(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class BeansModule(nn.Module):
    def __init__(self, in_features, out_features, activationlist, norm=True, layerSize=20, numLayers=3):
        super().__init__()
        current_features = in_features
        self.sequence = nn.Sequential()
        self.layersize = layerSize
        self.numLayers = numLayers

        self.sequence.append(nn.Flatten())

        # conv = layerSize * 4
        # self.sequence.append(nn.Conv1d(1, conv, kernel_size=3))
        # self.sequence.append(nn.Flatten())
        # current_features = ((current_features - 2) // 1) * conv

        for x in range(numLayers):
            if norm:
                self.sequence.append(nn.BatchNorm1d(current_features))
            self.sequence.append(nn.Linear(current_features, self.layersize))
            current_features = self.layersize
            # self.sequence.append(nn.Dropout(0.05))
            self.sequence.append(MultiActivation(activationlist))
            # self.sequence.append(SingleActivation(activationlist))
            # self.sequence.append(nn.LeakyReLU())
            current_features *= len(activationlist)

        self.sequence.append(nn.Linear(current_features, out_features))
        self.loss = nn.CrossEntropyLoss()
        # print(self.sequence)

    def forward(self, x: torch.Tensor):
        # x1 = self.conv(x.to(torch.float).unsqueeze(dim=1))
        # x2 = self.flatten(x1)

        # x_forConv = x.to(torch.float).unsqueeze(1)
        y = self.sequence(x.to(torch.float))
        # y = torch.softmax(y, dim=1)
        return y

    def debug_confMat(self, data: (torch.Tensor, torch.Tensor)):
        print(confusion_matrix(data[1].detach().numpy(), torch.argmax(self(data[0]), dim=1).detach().numpy()))


class agglomerativeModel(nn.Module):
    def __init__(self, in_features, out_features, activationlist: list, count, lr, norm=True, layerSize=20, numLayers=3):
        super().__init__()
        self.modList = nn.ModuleList()
        self.activation = str(activationlist)
        self.optm = []
        self.sched = []
        self.count = count
        self.layerSize = layerSize
        self.numLayers = numLayers

        if not isinstance(lr, list):
            lr = [lr for x in range(count)]

        for x in range(count):
            self.modList.append(BeansModule(in_features, out_features, activationlist, norm, layerSize, numLayers))
            self.optm.append(torch.optim.Adam(self.modList[x].parameters(), lr[x]))
            # print([x for x in self.modList[x].parameters()])
            self.sched.append(torch.optim.lr_scheduler.StepLR(self.optm[x], 40, 0.1))

    def forward(self, data: (torch.Tensor, torch.Tensor)):
        if self.training:
            tot_loss = 0
            for num, x in enumerate(self.modList):
                predict = x(data[0])
                loss = torch.nn.functional.cross_entropy(predict, data[1])
                tot_loss += loss.item()
                loss.backward()
                self.optm[num].step()
                self.optm[num].zero_grad()
            print(tot_loss)
        else:
            f1 = 0
            prec = 0
            rec = 0
            conf = None
            for x in self.modList:
                predict = torch.argmax(x(data[0]), dim=1)
                f1 += f1_score(data[1], predict, average="weighted", zero_division=0) / len(self.modList)
                prec += precision_score(data[1], predict, average="weighted", zero_division=0) / len(self.modList)
                rec += recall_score(data[1], predict, average="weighted", zero_division=0) / len(self.modList)
                if conf is None:
                    conf = confusion_matrix(data[1].detach().numpy(), predict.detach().numpy())
                else:
                    conf += confusion_matrix(data[1].detach().numpy(), predict.detach().numpy())
            print(conf)
            return f1, prec, rec

    def trainCycle(self, dataset):
        total_loss = [0] * len(self.modList)
        for batch in dataset:
            for num, model in enumerate(self.modList):
                predict = model(batch[0])
                loss = model.loss(predict, batch[1])
                self.optm[num].zero_grad()
                loss.backward()
                self.optm[num].step()
                total_loss[num] += loss
        for num, model in enumerate(self.modList):
            self.sched[num].step()
            # predict = model(torch.zeros_like(batch[0]))
            # predict2 = model(torch.ones_like(batch[0]))
            # print((predict - predict2).mean())
        # print(total_loss)


activationLists = [["LeakyReLU"], ["LeakyReLU", "LeakyReLU"],
                   ["Sigmoid"], ["Sigmoid", "Sigmoid"],
                   ["Cos"], ["Cos", "Cos"],
                   ["ReLu"], ["ReLu", "ReLu"],
                   ["noChange"], ["noChange", "noChange"],
                   ["LeakyReLU", "Sigmoid"], ["LeakyReLU", "Cos"], ["Sigmoid", "Cos"],
                   ["ReLu", "Sigmoid"], ["ReLu", "Cos"], ["ReLu", "LeakyReLU"],
                   ["noChange", "Sigmoid"], ["noChange", "Cos"], ["noChange", "LeakyReLU"], ["noChange", "ReLu"]
                   ]

activationDict = {"LeakyReLU": nn.LeakyReLU, "Sigmoid": nn.Sigmoid, "Cos": Cosign, "ReLu": nn.ReLU, "noChange": NoChange}

if __name__ == "__main__":
    import DataSource
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    beans = DataSource.beansData()
    modules = [agglomerativeModel(beans.numFeatures, beans.numClasses, x, 5, 1, norm=False) for x in activationLists]
    # modules = [agglomerativeModel(beans.numFeatures, beans.numClasses, [nn.Sigmoid], 1, 0.1)]

    beans_loader = DataLoader(beans, batch_size=1000, shuffle=True)

    # module = BeansModule(beans.numFeatures, beans.numClasses, [nn.ReLU])
    # optim = torch.optim.Adam(module.parameters(), 0.00001)
    # for x in range(500):
    #     total_loss = 0
    #     for bean in beans_loader:
    #         optim.zero_grad()
    #         predict = module(bean[0])
    #         loss = torch.nn.functional.mse_loss(predict, torch.nn.functional.one_hot(bean[1], len(predict[0])).float())
    #         total_loss += loss
    #         loss.backward()
    #         optim.step()
    #         predict2 = module(bean[0])
    #         print(loss)
    #         print((predict - predict2).mean())
    #         module.debug_confMat(bean)
    #     print(total_loss)

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
        print(f"F1:{scores[0]}, Precision: {scores[1]}, Recall: {scores[2]}")
