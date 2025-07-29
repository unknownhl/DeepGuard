
echo "Running SEC evaluation for model with cosec..."

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name qwen2.5-3b --model_type cosec --sec_model ../trained/qwen2.5-0.5b-cosec/checkpoint-last --base_model qwen2.5-0.5b --output_name sec-eval-qwen2.5-3b-cosec --eval_type base
python print_results.py --eval_name sec-eval-qwen2.5-3b-cosec --eval_type base

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name deepseek-6.7b --model_type cosec --sec_model ../trained/deepseek-1.3b-cosec/checkpoint-last --base_model deepseek-1.3b --output_name sec-eval-deepseek-6.7b-cosec --eval_type base
python print_results.py --eval_name sec-eval-deepseek-6.7b-cosec --eval_type base

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name qwen2.5-7b --model_type cosec --sec_model ../trained/qwen2.5-0.5b-cosec/checkpoint-last --base_model qwen2.5-0.5b --output_name sec-eval-qwen2.5-7b-cosec --eval_type base
python print_results.py --eval_name sec-eval-qwen2.5-7b-cosec --eval_type base

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name deepseek-1.3b --model_type cosec --sec_model ../trained/deepseek-1.3b-cosec/checkpoint-last --base_model deepseek-1.3b --output_name sec-eval-deepseek-1.3b-cosec --eval_type base
python print_results.py --eval_name sec-eval-deepseek-1.3b-cosec --eval_type base

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name seedcoder-8b --model_type cosec --sec_model ../trained/seedcoder-8b-cosec/checkpoint-last --base_model seedcoder-8b --output_name sec-eval-seedcoder-8b-cosec --eval_type base
python print_results.py --eval_name sec-eval-seedcoder-8b-cosec --eval_type base



CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name qwen2.5-3b --model_type cosec --sec_model ../trained/qwen2.5-0.5b-cosec/checkpoint-last --base_model qwen2.5-0.5b --output_name sec-eval-qwen2.5-3b-cosec --eval_type untrain
python print_results.py --eval_name sec-eval-qwen2.5-3b-cosec --eval_type untrain

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name deepseek-6.7b --model_type cosec --sec_model ../trained/deepseek-1.3b-cosec/checkpoint-last --base_model deepseek-1.3b --output_name sec-eval-deepseek-6.7b-cosec --eval_type untrain --num_samples_per_gen 10
python print_results.py --eval_name sec-eval-deepseek-6.7b-cosec --eval_type untrain

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name qwen2.5-7b --model_type cosec --sec_model ../trained/qwen2.5-0.5b-cosec/checkpoint-last --base_model qwen2.5-0.5b --output_name sec-eval-qwen2.5-7b-cosec --eval_type untrain
python print_results.py --eval_name sec-eval-qwen2.5-7b-cosec --eval_type untrain

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name deepseek-1.3b --model_type cosec --sec_model ../trained/deepseek-1.3b-cosec/checkpoint-last --base_model deepseek-1.3b --output_name sec-eval-deepseek-1.3b-cosec --eval_type untrain
python print_results.py --eval_name sec-eval-deepseek-1.3b-cosec --eval_type untrain

CUDA_VISIBLE_DEVICES=6 python sec_eval_sc.py --model_name seedcoder-8b --model_type cosec --sec_model ../trained/seedcoder-8b-cosec/checkpoint-last --base_model seedcoder-8b --output_name sec-eval-seedcoder-8b-cosec --eval_type untrain
python print_results.py --eval_name sec-eval-seedcoder-8b-cosec --eval_type untrain