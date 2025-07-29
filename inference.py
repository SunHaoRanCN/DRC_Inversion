import argparse
from omegaconf import OmegaConf
from run.train import classifier_train, regressor_train
from run.evaluation import classifier_eval, regressor_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        default="classification",
        help="classification or regression task",
    )
    parser.add_argument(
        "-m",
        "--mission",
        type=str,
        default="train",
        help="train or evaluation",
    )
    parser.add_argument(
        "-i",
        "--input_path",
        type=str,
        help="Folder to the compressed signals"
    )
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default='./outputs',
        help='Folder to save the decompressed signals',
    )
    parser.add_argument(
        "--bs",
        type=int,
        default=8,
        help='Batch size',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help='Random Seed',
    )
    parser.add_argument(
        "--config",
        type=str,
        default="conf/config.yaml",
        help='configuration',
    )
    args = parser.parse_args()

    if args.task == 'classification':
        cfg_path = 'conf/class.yaml'
    elif args.task == 'restoration':
        cfg_path = 'configs/reg.yaml'
    else:
        raise ValueError("Only accept task types of 'classification' and 'regression'!")

    configs = OmegaConf.load(cfg_path)
    configs.seed = args.seed

    if args.task == 'classification':
        if args.mission == 'train':
            classifier_train(configs)
        elif args.mission == "evaluation":
            classifier_eval(configs)
        else:
            ValueError("Only accept mission types of 'train' and 'evaluation'!")

    elif args.task == 'regression':
        if args.mission == 'train':
            regressor_train(configs)
        elif args.mission == "evaluation":
            regressor_eval(configs)
        else:
            ValueError("Only accept mission types of 'train' and 'evaluation'!")

    else:
        raise ValueError("Only accept task types of 'classification' and 'regression'!")

if __name__ == '__main__':
    main()
