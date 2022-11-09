import argparse
from src.config import Config
from src.utils.data_process import *
from src.model.Metagraph2vec import *
from src.utils.sampler import *
from src.utils.hete_random_walk import *
from src.utils.utils import *
import warnings
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings('ignore')

seed = 0
def main():
    args = init_params()
    config_file = ["./src/config.ini"]
    config = Config(config_file, args)

    g_hin = HIN(config.input_fold, config.data_type, config.relation_list)

    config.temp_file += 'graph_rw.txt'
    config.out_emd_file += args.dataset + '_node.txt'
    mgg = MetaGraphGenerator()
    if args.dataset == "acm":
        mgg.generate_random_three(config.temp_file, config.num_walks, config.walk_length, g_hin.node,
                                    g_hin.relation_dict)
    elif args.dataset == "dblp":
        mgg.generate_random_four(config.temp_file, config.num_walks, config.walk_length, g_hin.node,
                                    g_hin.relation_dict)
    model = Metagraph2VecTrainer(config,g_hin)
    print("Training")
    model.train()


def init_params():
    parser = argparse.ArgumentParser(description="OPEN-HINE")
    parser.add_argument('-d', '--dataset', default='dblp', type=str, help="Dataset")
    parser.add_argument('-m', '--model', default='MetaGraph2vec', type=str, help='Train model')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
