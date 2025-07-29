# enhanced_inference.py
import os
import torch
from transformers import AutoTokenizer
from deepguard.train import SecurityAwareLoRAModel



def load_model(model_type, path, is_training, args):
    if model_type == 'deepguard':
        if not is_training:
            from sven.constant import MODEL_DIRS
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            tokenizer = AutoTokenizer.from_pretrained(
                MODEL_DIRS[args.model_name],
                cache_dir=cache_dir,
                local_files_only=True
            )
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token_id = tokenizer.bos_token_id
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            possible_paths = [
                os.path.join(path, 'checkpoint-best'),
                path]
            
            checkpoint_path = None
            for path_candidate in possible_paths:
                if (os.path.exists(os.path.join(path_candidate, 'training_state.pt'))):
                    checkpoint_path = path_candidate
                    break
            
            if checkpoint_path is None:
                raise FileNotFoundError(f"No valid checkpoint found in any of: {possible_paths}")
            
            print(f"Loading enhanced checkpoint from: {checkpoint_path}")
            training_state = torch.load(
                os.path.join(checkpoint_path, 'training_state.pt'),
                map_location=device
            )
            base_model_path = training_state['model_path']

            model = SecurityAwareLoRAModel.from_pretrained(
                checkpoint_path,
                base_model_path,
                tokenizer,
                device=device
            )
            return tokenizer, model, device
            
    else:
        raise ValueError(f"Unsupported model type: {model_type}")