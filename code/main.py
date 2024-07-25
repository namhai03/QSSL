import train
import eval

def run_experiment(seed, classifier, nlayers, gamma, data, c1, c2, device):
    _k = [1, 2, 5, 10]

    print(f"Running training with seed={seed}, device={device}")
    print(f"Classifier: {classifier}, Layers: {nlayers}, Gamma: {gamma}")
    print(f"Dataset: {data}, c1: {c1}, c2: {c2}")

    train.main(seed, classifier, nlayers, gamma, data, c1, c2, device)

    for k in _k:
        print(f"Evaluating k={k}")
        eval.main(seed, classifier, nlayers, gamma, data, k, c1, c2, device)
