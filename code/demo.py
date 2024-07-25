import argparse
from random import randrange
import main_v1

def parse_arguments():
    parser = argparse.ArgumentParser(description= "Run training and validation")

    # select device
    parser.add_argument(
        "--device",
        type= str,
        choices= ["cpu", "cuda:0"],
        default= "cuda:0",
        help= "Choose device cpu or cuda:0. Default is cuda:0"
    )

    # seed 
    parser.add_argument(
        "--seed",
        type= int,
        default= None,
        help= "Choose seed. Default is random seed"
    )

    # Select classifier: classical or quantum
    parser.add_argument(
        "--classifier",
        choices=["classical", "quantum"],
        default="quantum",
        help= "Choose classifier, quantum or classical. Default is quantum"
    )

    # Number of layers 
    parser.add_argument(
        "--nlayers",
        type= int,
        default= 2,
        help= "Number of layers. Default is 2"
    )

    # Gamma contribution for KLL Loss
    parser.add_argument(
        "--gamma",
        type= float,
        default=0.5,
        help= "Gamma value is in range (0,1). Default is 0.5"
    )

    # Data 
    parser.add_argument(
        "--data",
        type= str, 
        choices= ["MNIST", "FashionMNIST", "KMNIST"],
        default= "MNIST",
        help= "Choose dataset: MNIST, FashionMNIST, KMNIST. Default is MNIST"
    )

    # class for target/label c1
    parser.add_argument(
        "--c1",
        type=int,
        default=0,
        help="Class label 1 (0-9). Default is 0.",
    )

    # Class 2 argument with default value and validation
    parser.add_argument(
        "--c2",
        type=int,
        default=1,
        help="Class label 2 (0-9) and different from c1. Default is 1.",
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    seed = args.seed if args.seed is not None else randrange(200)

    if not (0 < args.gamma < 1):
        raise ValueError("Choose gamma between (0,1)")
    
    if not (0 <= args.c1 <= 9):
        raise ValueError("Choose the target/label for c1 between 0-9")
    
    if not (0 <= args.c2 <= 9):
        raise ValueError("Choose the target/label for c1 between 0-9")
    
    if args.c1 == args.c2:
        raise ValueError("Choose a different label for c2 than c1")
    
    main_v1.run_experiment(
        seed=seed,
        classifier=args.classifier,
        nlayers=args.nlayers,
        gamma=args.gamma,
        data=args.data,
        c1=args.c1,
        c2=args.c2,
        device=args.device
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")