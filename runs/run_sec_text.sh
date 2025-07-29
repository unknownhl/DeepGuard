
echo "Running SEC evaluation for text model..."

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name qwen2.5-3b --model_type text --model_dir qwen2.5-3b --output_name sec-eval-qwen2.5-3b-text --eval_type base
python print_results.py --eval_name sec-eval-qwen2.5-3b-text --eval_type base

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name qwen2.5-7b --model_type text --model_dir qwen2.5-7b --output_name sec-eval-qwen2.5-7b-text --eval_type base
python print_results.py --eval_name sec-eval-qwen2.5-7b-text --eval_type base

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name deepseek-6.7b --model_type text --model_dir deepseek-6.7b --output_name sec-eval-deepseek-6.7b-text --eval_type base
python print_results.py --eval_name sec-eval-deepseek-6.7b-text --eval_type base

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name deepseek-1.3b --model_type text --model_dir deepseek-1.3b --output_name sec-eval-deepseek-1.3b-text --eval_type base
python print_results.py --eval_name sec-eval-deepseek-1.3b-text --eval_type base

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name seedcoder-8b --model_type text --model_dir seedcoder-8b --output_name sec-eval-seedcoder-8b-text --eval_type base
python print_results.py --eval_name sec-eval-seedcoder-8b-text --eval_type base


CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name qwen2.5-3b --model_type text --model_dir qwen2.5-3b --output_name sec-eval-qwen2.5-3b-text --eval_type untrain
python print_results.py --eval_name sec-eval-qwen2.5-3b-text --eval_type untrain

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name qwen2.5-7b --model_type text --model_dir qwen2.5-7b --output_name sec-eval-qwen2.5-7b-text --eval_type untrain
python print_results.py --eval_name sec-eval-qwen2.5-7b-text --eval_type untrain

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name deepseek-6.7b --model_type text --model_dir deepseek-6.7b --output_name sec-eval-deepseek-6.7b-text --eval_type untrain
python print_results.py --eval_name sec-eval-deepseek-6.7b-text --eval_type untrain

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name deepseek-1.3b --model_type text --model_dir deepseek-1.3b --output_name sec-eval-deepseek-1.3b-text --eval_type untrain
python print_results.py --eval_name sec-eval-deepseek-1.3b-text --eval_type untrain

CUDA_VISIBLE_DEVICES=7 python sec_eval_sc.py --model_name seedcoder-8b --model_type text --model_dir seedcoder-8b --output_name sec-eval-seedcoder-8b-text --eval_type untrain
python print_results.py --eval_name sec-eval-seedcoder-8b-text --eval_type untrain
