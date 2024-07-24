import torch
import torch.nn as nn
import pennylane as qml
from einops import rearrange
import torch.nn.functional as F

class ClassicalClassifier(nn.Module):
    def __init__(self, nlayers, device):
        super(ClassicalClassifier, self).__init__()
        self.cls_conv = nn.Conv2d(1, 1, (7,7), stride = 5)
        self.bn = nn.BatchNorm2d(1)
        cls_layers = [nn.Linear(4,4) for _ in range(nlayers)]
        self.cls_layers = nn.Sequential(*cls_layers)
        self.mlp = nn.Linear(4,2)

    def forward(self, x):
        x = self.cls_conv(x)
        x = self.bn(x)
        x = rearrange(x.detach(), 'b c h w -> b (c h w)')
        x = self.cls_layers(x)
        x = self.mlp(x.detach())
        x = F.log_softmax(x, dim = 1)
        return x


class QuantumClassifier(nn.Module):
    def __init__(self, nlayers, device):
        super(QuantumClassifier, self).__init__()
        self.cls_conv = nn.Conv2d(1, 1, (7,7), stride = 5)
        self.bn = nn.BatchNorm2d(1)
        self.qnn_conv = self.get_qlayer(nlayers)
        torch.nn.init.xavier_uniform_(self.cls_conv.weight)

    def forward(self, x):
        x = self.cls_conv(x)
        x = self.bn(x)
        x = rearrange(x.detach(), 'b c h w -> b (c h w)')
        x = self.qnn_conv(x)
        x = F.log_softmax(x, dim = 1)
        return x

    def get_qlayer(self, nlayers):
        n_qubits = 5
        dev = qml.device('default.qubit', wires = n_qubits)
        @qml.qnode(dev)
        def qnode(inputs, w_1, w_2, w_3):
            qml.AngleEmbedding(inputs, wires = range(n_qubits))
            qml.BasicEntanglerLayers(w_1, wires = [0,1])
            qml.BasicEntanglerLayers(w_2, wires = [2,3])
            qml.BasicEntanglerLayers(w_3, wires = [1,2])
            for i in range(4):
                qml.CNOT(wires = [i,4])
            return qml.probs(wires = 4)

        weight_shape = {"w_1": (nlayers,2),
                        "w_2": (nlayers,2),
                        "w_3": (nlayers,2)}
        qlayer = qml.qnn.TorchLayer(qnode, weight_shape)
        return qlayer
