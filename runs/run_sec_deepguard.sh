
echo "Running SEC evaluation for model with deepguard..."

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name qwen2.5-3b --model_type deepguard --model_dir ../trained/qwen2.5-3b-deepguard-attention --output_name sec-eval-qwen2.5-3b-deepguard --eval_type base
python print_results.py --eval_name sec-eval-qwen2.5-3b-deepguard --eval_type base

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name qwen2.5-7b --model_type deepguard --model_dir ../trained/qwen2.5-7b-deepguard-attention --output_name sec-eval-qwen2.5-7b-deepguard --eval_type base
python print_results.py --eval_name sec-eval-qwen2.5-7b-deepguard --eval_type base

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name deepseek-6.7b --model_type deepguard --model_dir ../trained/deepseek-6.7b-deepguard-attention --output_name sec-eval-deepseek-6.7b-deepguard --eval_type base
python print_results.py --eval_name sec-eval-deepseek-6.7b-deepguard --eval_type base

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name deepseek-1.3b --model_type deepguard --model_dir ../trained/deepseek-1.3b-deepguard-attention --output_name sec-eval-deepseek-1.3b-deepguard --eval_type base
python print_results.py --eval_name sec-eval-deepseek-1.3b-deepguard --eval_type base

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name seedcoder-8b --model_type deepguard --model_dir ../trained/seedcoder-8b-deepguard-attention --output_name sec-eval-seedcoder-8b-deepguard --eval_type base
python print_results.py --eval_name sec-eval-seedcoder-8b-deepguard --eval_type base


CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name qwen2.5-3b --model_type deepguard --model_dir ../trained/qwen2.5-3b-deepguard-attention --output_name sec-eval-qwen2.5-3b-deepguard --eval_type untrain
python print_results.py --eval_name sec-eval-qwen2.5-3b-deepguard --eval_type untrain

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name qwen2.5-7b --model_type deepguard --model_dir ../trained/qwen2.5-7b-deepguard-attention --output_name sec-eval-qwen2.5-7b-deepguard --eval_type untrain
python print_results.py --eval_name sec-eval-qwen2.5-7b-deepguard --eval_type untrain

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name deepseek-6.7b --model_type deepguard --model_dir ../trained/deepseek-6.7b-deepguard-attention --output_name sec-eval-deepseek-6.7b-deepguard --eval_type untrain
python print_results.py --eval_name sec-eval-deepseek-6.7b-deepguard --eval_type untrain

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name deepseek-1.3b --model_type deepguard --model_dir ../trained/deepseek-1.3b-deepguard-attention --output_name sec-eval-deepseek-1.3b-deepguard --eval_type untrain
python print_results.py --eval_name sec-eval-deepseek-1.3b-deepguard --eval_type untrain

CUDA_VISIBLE_DEVICES=7 python sec_eval.py --model_name seedcoder-8b --model_type deepguard --model_dir ../trained/seedcoder-8b-deepguard-attention --output_name sec-eval-seedcoder-8b-deepguard --eval_type untrain
python print_results.py --eval_name sec-eval-seedcoder-8b-deepguard --eval_type untrain


