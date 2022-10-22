import argparse
import time
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_parser():
    parser = argparse.ArgumentParser(description='k-GNN experiment.')

    # General
    parser.add_argument('--seed', type=int, default=43,
                        help='random seed to set (default: 43, i.e. the non-meaning of life))')
    parser.add_argument('--no-train', default=False)

    # Dataset config
    parser.add_argument('--dataset_name', type=str, default="TU_MUTAG",
                        help='dataset name (default: TU_MUTAG)')
    parser.add_argument('--data_format', type=str, default="PyG",
                        help='dataset format (default: PyG)')
    parser.add_argument('--task_type', type=str, default='classification',
                        help='task type, either (bin)classification, regression or isomorphism (default: classification)')
    parser.add_argument('--eval_metric', type=str, default='accuracy',
                        help='evaluation metric (default: accuracy)')
    parser.add_argument('--result_folder', type=str, default=os.path.join(ROOT_DIR, 'exp', 'results'),
                        help='filename to output result (default: None, will use `scn/exp/results`)')

    # Training & Evaluation
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--folds', type=int, default=10,
                        help='The number of folds to run on in cross validation experiments')

    # NN design
    parser.add_argument('--nonlinearity', type=str, default='elu',
                        help='activation function (default: relu)')

    # GNN design
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in models (default: 64)')
    parser.add_argument('--pool_func', type=str, default='ave_pool',
                        help='Which graph pooling function to use (possible arguments: ave_pool, max_pool, add_pool)')
    parser.add_argument('--num_linear_layers', type=int, default=3,
                        help='number of linear layers before the prediction (default: 3.')

    # Method-specific hyperparameters
    parser.add_argument('--num_layers_per_dim', type=int, default=3,
                        help='number of message passing layers (default: 3)')
    parser.add_argument('--initial_emb_dim', type=int, default=32,
                        help='dimensionality of hidden units in models (default: 32)')
    parser.add_argument('--max_k', type=int, default=3,
                        help='Maximum length of node-tuples (default: 3)')

    # Regularisation
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='dropout rate (default: 0.0)')

    # Optimiser
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                        help='strength of lr decay (default: 0.5)')
    parser.add_argument('--lr_scheduler_patience', type=float, default=5,
                        help='patience for `ReduceLROnPlateau` lr decay (default: 10)')
    parser.add_argument('--lr_scheduler_min', type=float, default=0.00001,
                        help='min LR for `ReduceLROnPlateau` lr decay (default: 1e-5)')
    parser.add_argument('--lr_scheduler_step_size', type=float, default=30,
                        help='Steps after which to reduce the LR: (default: 30)')



    return parser