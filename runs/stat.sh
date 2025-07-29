model=seedcoder-8b
python correctness_eval.py --paths ../experiments/sec_eval/sec-eval-$model-deepguard --do_eval --num_seeds 1 --eval_type base
python new_stats.py --paths ../experiments/sec_eval/sec-eval-$model-deepguard --eval_type base 
python correctness_eval.py --paths ../experiments/sec_eval/sec-eval-$model-deepguard --do_print --num_seeds 1 --eval_type base
