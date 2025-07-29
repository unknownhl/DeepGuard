import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(project_root)
import json
import argparse

from sven.metric import SecEval

EVAL_CHOICES = [
    'base',
    'untrain',
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_name', type=str, required=True)
    parser.add_argument('--detail', action='store_true', default=False)
    parser.add_argument('--eval_type', type=str, choices=EVAL_CHOICES, default='base')
    parser.add_argument('--split', type=str, choices=['val', 'test', 'all', 'validation', 'intersec', 'diff'], default='test')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_shots', type=int, default=5)
    parser.add_argument('--experiments_dir', type=str, default='../experiments')
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    e = SecEval(os.path.join(args.experiments_dir, 'sec_eval', args.eval_name), args.split, args.eval_type)
    e.pretty_print(args.detail)

if __name__ == '__main__':
    main()