import argparse
import os

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
        help="Folder to the processed signals"
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
    args = parser.parse_args()

    if args.task == 'classification':
        cfg_path = 'conf/conf_class.yaml'
    elif args.task == 'regression':
        cfg_path = 'conf/conf_reg.yaml'
    else:
        raise ValueError("Only accept task types of 'classification' and 'regression'!")

    configs = OmegaConf.load(cfg_path)
    configs.seed = args.seed

    if args.task == 'classification':
        os.makedirs("../experiments", exist_ok=True)
        if args.mission == 'train':
            ct = classifier_train(configs, args.input_path, if_augmentation=configs.if_augmentation)
            ct.train()
        elif args.mission == "evaluation":
            ce = classifier_eval(configs, args.input_path, args.out_path)
            ce.evaluate()
        else:
            raise ValueError("Only accept mission types of 'train' and 'evaluation'!")

    elif args.task == 'regression':
        if args.mission == 'train':
            rt = regressor_train(configs, args.input_path, if_augmentation=configs.if_augmentation)
            rt.train()
        elif args.mission == "evaluation":
            re = regressor_eval(configs, args.input_path, args.out_path)
            re.evaluate()
        else:
            raise ValueError("Only accept mission types of 'train' and 'evaluation'!")

    else:
        raise ValueError("Only accept task types of 'classification' and 'regression'!")

if __name__ == '__main__':
    main()
