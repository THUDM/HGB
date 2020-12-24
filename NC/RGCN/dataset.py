import argparse
from networkx.algorithms.assortativity.correlation import numeric_ac
import torch as th
from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
import dgl


def main(args):
    # load graph data
    ogb_dataset = False
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()

    # Load from hetero-graph
    hg = dataset[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    args = parser.parse_args()
    print(args)
    main(args)
