import train
import eval
from random import randrange


device = 'cpu'

for _ in range(50):
    seed = randrange(200)
    #_classifier = ['quantum']
    _classifier = ['classical']
    _nlayers = [2]
    _gamma = [0.5]
    #_data = ['MNIST','FashionMNIST','KMNIST']
    _data = ['MNIST']
    _k = [1,2,5,10]
    _c1 = [0]
    _c2 = [1]
    for classifier in _classifier:
        for data in _data:
            for nlayers in _nlayers:
                for gamma in _gamma:
                    for c1 in _c1:
                        for c2 in _c2:
                            print(classifier, nlayers, gamma, data, c1, c2)
                            train.main(seed, classifier, nlayers, gamma, data, c1, c2, device)
                            for k in _k:
                                print(classifier, nlayers, gamma, data, c1, c2, k)
                                eval.main(seed, classifier, nlayers, gamma, data, k, c1, c2, device)
