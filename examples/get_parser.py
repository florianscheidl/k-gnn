import argparse
import time
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_parser():
    parser = argparse.ArgumentParser(description='k-GNN experiment.')
    parser.add_argument('--no-train', default=False)
    parser.add_argument('--seed', type=int, default=43,
                        help='random seed to set (default: 43, i.e. the non-meaning of life))')
    # parser.add_argument('--start_seed', type=int, default=0,
    #                     help='The initial seed when evaluating on multiple seeds.')
    # parser.add_argument('--stop_seed', type=int, default=9,
    #                     help='The final seed when evaluating on multiple seeds.')
    # parser.add_argument('--device', type=int, default=0,
    #                     help='which gpu to use if any (default: 0)')
    # parser.add_argument('--model', type=str, default='sparse_cin',
    #                     help='model, possible choices: cin, dummy, ... (default: cin)')
    parser.add_argument('--drop_rate', type=float, default=0.0,
                        help='dropout rate (default: 0.0)')
    parser.add_argument('--nonlinearity', type=str, default='elu',
                        help='activation function (default: relu)')
    # parser.add_argument('--pool_func', type=str, default='sum',
    #                     help='Pooling function (default: avg_pool)')
   # parser.add_argument('--graph_norm', type=str, default='bn', choices=['bn', 'ln', 'id'],
   #                      help='Normalization layer to use inside the model')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    # parser.add_argument('--lr_scheduler', type=str, default='StepLR',
    #                     help='learning rate decay scheduler (default: StepLR)')
    # parser.add_argument('--lr_scheduler_decay_steps', type=int, default=50,
    #                     help='number of epochs between lr decay (default: 50)')
    parser.add_argument('--lr_scheduler_decay_rate', type=float, default=0.5,
                        help='strength of lr decay (default: 0.5)')
    parser.add_argument('--lr_scheduler_patience', type=float, default=5,
                        help='patience for `ReduceLROnPlateau` lr decay (default: 10)')
    parser.add_argument('--lr_scheduler_min', type=float, default=0.00001,
                        help='min LR for `ReduceLROnPlateau` lr decay (default: 1e-5)')
    parser.add_argument('--max_k', type=int, default=3,
                        help='Maximum length of node-tuples')
    parser.add_argument('--num_layers_per_dim', type=int, default=3,
                        help='number of message passing layers (default: 5)')
    parser.add_argument('--num_linear_layers', type=int, default=3,
                        help='number of linear layers before the prediction.')
    parser.add_argument('--initial_emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in models (default: 300)')
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='dimensionality of hidden units in models (default: 300)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    # parser.add_argument('--num_workers', type=int, default=0,
    #                     help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="TU_MUTAG",
                        help='dataset name (default: PROTEINS)')

    # TODO
    parser.add_argument('--task_type', type=str, default='classification',
                        help='task type, either (bin)classification, regression or isomorphism (default: classification)')

    # TODO
    parser.add_argument('--eval_metric', type=str, default='accuracy',
                        help='evaluation metric (default: accuracy)')
    # parser.add_argument('--iso_eps', type=int, default=0.01,
    #                     help='Threshold to define (non-)isomorphism')
    # parser.add_argument('--minimize', action='store_true',
    #                     help='whether to minimize evaluation metric or not')

    # TODO
    parser.add_argument('--result_folder', type=str, default=os.path.join(ROOT_DIR, 'exp', 'results'),
                        help='filename to output result (default: None, will use `scn/exp/results`)')
    # parser.add_argument('--exp_name', type=str, default=str(time.time()),
    #                     help='name for specific experiment; if not provided, a name based on unix timestamp will be '+\
    #                     'used. (default: None)')
    parser.add_argument('--folds', type=int, default=10,
                        help='The number of folds to run on in cross validation experiments')
    # parser.add_argument('--train_eval_period', type=int, default=10,
    #                     help='How often to evaluate on train.')
    # parser.add_argument('--use_edge_features', action='store_true',
    #                     help="Use edge features for molecular graphs")
    return parser