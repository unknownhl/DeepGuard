
echo "Running SEC evaluation for model with prefix tuning..."

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name qwen2.5-3b --model_type prefix --model_dir ../trained/qwen2.5-3b-prefix/checkpoint-last --output_name sec-eval-qwen2.5-3b-prefix --eval_type base
python print_results.py --eval_name sec-eval-qwen2.5-3b-prefix --eval_type base

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name deepseek-6.7b --model_type prefix --model_dir ../trained/deepseek-6.7b-prefix/checkpoint-last --output_name sec-eval-deepseek-6.7b-prefix --eval_type base
python print_results.py --eval_name sec-eval-deepseek-6.7b-prefix --eval_type base

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name qwen2.5-7b --model_type prefix --model_dir ../trained/qwen2.5-7b-prefix/checkpoint-last --output_name sec-eval-qwen2.5-7b-prefix --eval_type base
python print_results.py --eval_name sec-eval-qwen2.5-7b-prefix --eval_type base

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name deepseek-1.3b --model_type prefix --model_dir ../trained/deepseek-1.3b-prefix/checkpoint-last --output_name sec-eval-deepseek-1.3b-prefix --eval_type base
python print_results.py --eval_name sec-eval-deepseek-1.3b-prefix --eval_type base

CUDA_VISIBLE_DEVICES=1 python sec_eval_sc.py --model_name seedcoder-8b --model_type prefix --model_dir ../trained/seedcoder-8b-prefix/checkpoint-last --output_name sec-eval-seedcoder-8b-prefix --eval_type base
python print_results.py --eval_name sec-eval-seedcoder-8b-prefix --eval_type base


CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name qwen2.5-3b --model_type prefix --model_dir ../trained/qwen2.5-3b-prefix/checkpoint-last --output_name sec-eval-qwen2.5-3b-prefix --eval_type untrain
python print_results.py --eval_name sec-eval-qwen2.5-3b-prefix --eval_type untrain

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name deepseek-6.7b --model_type prefix --model_dir ../trained/deepseek-6.7b-prefix/checkpoint-last --output_name sec-eval-deepseek-6.7b-prefix --eval_type untrain
python print_results.py --eval_name sec-eval-deepseek-6.7b-prefix --eval_type untrain

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name qwen2.5-7b --model_type prefix --model_dir ../trained/qwen2.5-7b-prefix/checkpoint-last --output_name sec-eval-qwen2.5-7b-prefix --eval_type untrain
python print_results.py --eval_name sec-eval-qwen2.5-7b-prefix --eval_type untrain

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name deepseek-1.3b --model_type prefix --model_dir ../trained/deepseek-1.3b-prefix/checkpoint-last --output_name sec-eval-deepseek-1.3b-prefix --eval_type untrain
python print_results.py --eval_name sec-eval-deepseek-1.3b-prefix --eval_type untrain

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name seedcoder-8b --model_type prefix --model_dir ../trained/seedcoder-8b-prefix/checkpoint-last --output_name sec-eval-seedcoder-8b-prefix --eval_type untrain
python print_results.py --eval_name sec-eval-seedcoder-8b-prefix --eval_type untrain
