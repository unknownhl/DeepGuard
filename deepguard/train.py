import os
import random
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import logging
import math
from collections import defaultdict

class MultiLayerAggregator(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, aggregation_method: str = 'attention'):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.aggregation_method = aggregation_method

        if aggregation_method == 'attention':
            self.query = nn.Linear(hidden_size, hidden_size)
            self.key = nn.Linear(hidden_size, hidden_size)
            self.value = nn.Linear(hidden_size, hidden_size)
            self.output_proj = nn.Linear(hidden_size, hidden_size)
        elif aggregation_method in ['average', 'last_layer']:
            pass
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    def forward(self, hidden_states_list: List[torch.Tensor]) -> torch.Tensor:
        if self.aggregation_method == 'last_layer':
            return hidden_states_list[-1]
        
        elif self.aggregation_method == 'average':
            stacked = torch.stack(hidden_states_list, dim=0)
            return stacked.mean(dim=0)

        elif self.aggregation_method == 'attention':
            stacked = torch.stack(hidden_states_list, dim=1)
            batch_size, num_layers, seq_len, hidden_size = stacked.shape
            
            stacked = stacked.permute(0, 2, 1, 3).reshape(-1, num_layers, hidden_size)
            
            # Use the mean as query
            query = self.query(stacked.mean(dim=1, keepdim=True))
            key = self.key(stacked)  
            value = self.value(stacked) 
            
            scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(hidden_size)
            attn_weights = F.softmax(scores, dim=-1)
            
            attended = torch.matmul(attn_weights, value).squeeze(1)
            attended = self.output_proj(attended)
            return attended.reshape(batch_size, seq_len, hidden_size)
        
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

class SecurityAnalyzer(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int = 4, 
                 embed_dim: int = 128, aggregation_method: str = 'attention'):
        super().__init__()
        
        # Token security embeddings
        self.security_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.aggregation_method = aggregation_method
        self.num_layers = num_layers
    
        # Multi-layer aggregator
        self.layer_aggregator = MultiLayerAggregator(
            num_layers=num_layers,
            hidden_size=hidden_size,
            aggregation_method=aggregation_method
        )
        
        self.context_processor = nn.Sequential(
            nn.Linear(hidden_size + embed_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize
        nn.init.normal_(self.security_embeddings.weight, std=0.02)
        for m in self.context_processor.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, hidden_states_list: List[torch.Tensor], input_ids: torch.Tensor) -> torch.Tensor:
        if self.aggregation_method == 'last_layer' and len(hidden_states_list) > 1:
            hidden_states_list = [hidden_states_list[-1]]
        aggregated_hidden = self.layer_aggregator(hidden_states_list)
        security_embeds = self.security_embeddings(input_ids)
        combined = torch.cat([aggregated_hidden, security_embeds], dim=-1)
        scores = self.context_processor(combined).squeeze(-1)
        return scores


class SecurityAwareLoRAModel(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 tokenizer: AutoTokenizer,
                 target_modules: Optional[List[str]] = None,
                 device: Union[str, torch.device] = "cpu",
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 num_hidden_layers_to_use: int = 4,  
                 aggregation_method: str = 'attention',
                 logger: Optional[logging.Logger] = None):
        super().__init__()
        
        self.tokenizer = tokenizer
        self.device = device
        self.vocab_size = model.config.vocab_size
        self.hidden_size = model.config.hidden_size
        self.total_layers = model.config.num_hidden_layers
        self.num_hidden_layers_to_use = min(num_hidden_layers_to_use, self.total_layers)
        self.aggregation_method = aggregation_method
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.logger = logger or logging.getLogger(__name__)

        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            model.config.use_cache = False 
        
        if target_modules is None:
            if hasattr(model.config, 'model_type'):
                if 'llama' in model.config.model_type.lower():
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                elif 'qwen' in model.config.model_type.lower():
                    target_modules = ["q_proj", "v_proj"]
                elif 'gpt' in model.config.model_type.lower():
                    target_modules = ["c_attn"]
                else:
                    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
                
        
        self.target_modules = target_modules
        self.logger.info(f"Target modules for LoRA: {target_modules}")
        self.logger.info(f"Using top {self.num_hidden_layers_to_use} layers with {aggregation_method} aggregation")
        
        # Create LoRA config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(model, peft_config)
        self.model.print_trainable_parameters()

        self.security_analyzer = SecurityAnalyzer(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_hidden_layers_to_use,
            embed_dim=128,
            aggregation_method=aggregation_method
        ).to(device)

        self.feature_aggregator = MultiLayerAggregator(
            num_layers=self.num_hidden_layers_to_use,
            hidden_size=self.hidden_size,
            aggregation_method=aggregation_method
        ).to(device)
        
        self.register_buffer('token_security_stats', torch.zeros(self.vocab_size))
    
    def get_selected_hidden_states(self, all_hidden_states: Tuple[torch.Tensor]) -> List[torch.Tensor]:
        selected_layers = all_hidden_states[-(self.num_hidden_layers_to_use):]
        return list(selected_layers)
    
    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        loss_dict = {}
        labels_secure = batch['input_ids_secure'].clone()
        labels_secure[labels_secure == self.tokenizer.pad_token_id] = -100
        
        outputs_secure = self.model(
            input_ids=batch['input_ids_secure'],
            attention_mask=batch['attention_mask_secure'],
            labels=labels_secure,
            output_hidden_states=True,
            return_dict=True
        )
        
        generation_loss = outputs_secure.loss
        loss_dict['generation'] = generation_loss.item()
        
        with torch.no_grad():
            outputs_vulnerable = self.model(
                input_ids=batch['input_ids_vulnerable'],
                attention_mask=batch['attention_mask_vulnerable'],
                output_hidden_states=True,
                return_dict=True
            )
        hidden_states_secure = self.get_selected_hidden_states(outputs_secure.hidden_states)
        hidden_states_vulnerable = self.get_selected_hidden_states(outputs_vulnerable.hidden_states)

        del outputs_vulnerable
        torch.cuda.empty_cache()
    
        scores_secure = self.security_analyzer(
            hidden_states_secure, 
            batch['input_ids_secure']
        )
        scores_vulnerable = self.security_analyzer(
            hidden_states_vulnerable,
            batch['input_ids_vulnerable']
        )

        del hidden_states_secure, hidden_states_vulnerable
        torch.cuda.empty_cache()

        security_loss = F.relu(0.5 - (scores_secure.mean() - scores_vulnerable.mean()))
        loss_dict['security'] = security_loss.item()

        del scores_secure, scores_vulnerable
        torch.cuda.empty_cache()  
        
        kl_loss = 0
        if hasattr(self, 'use_kl_loss') and self.use_kl_loss:
            self.model.disable_adapter_layers()
            with torch.no_grad():
                ref_outputs = self.model(
                    input_ids=batch['input_ids_secure'],
                    attention_mask=batch['attention_mask_secure'],
                    output_hidden_states=False
                )
                ref_probs = F.softmax(ref_outputs.logits, dim=-1)
            self.model.enable_adapter_layers()
            
            current_probs = F.softmax(outputs_secure.logits, dim=-1)

            del ref_outputs, outputs_secure
            torch.cuda.empty_cache()
            
            kl_div = torch.sum(
                ref_probs * (torch.log(ref_probs + 1e-8) - torch.log(current_probs + 1e-8)),
                dim=-1
            )

            valid_mask = (batch['input_ids_secure'] != self.tokenizer.pad_token_id).float()
            kl_loss = (kl_div * valid_mask).sum() / valid_mask.sum()
            
            kl_loss = torch.clamp(kl_loss, max=1.0)
            
            loss_dict['kl'] = kl_loss.item()
            
            del ref_probs, current_probs, kl_div, valid_mask
            torch.cuda.empty_cache()
        
        total_loss = generation_loss
        total_loss += 0.5 * security_loss
        
        if kl_loss > 0:
            total_loss += 1.0 * kl_loss
        
        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def generate_with_security(self, input_ids, attention_mask=None, **kwargs):
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            hidden_states = self.get_selected_hidden_states(outputs.hidden_states)
            security_scores = self.security_analyzer(hidden_states, input_ids)
            avg_security = security_scores.mean().item()
            adjustment_strength = 1 / avg_security
            normalized_stats = self.token_security_stats / (self.token_security_stats.abs().max() + 1e-8)
            vocab_adjustment = normalized_stats.unsqueeze(0) * adjustment_strength

        initial_length = input_ids.shape[1]
        
        def security_logits_processor(generated_ids, scores):
            current_length = generated_ids.shape[1]
            position = current_length - initial_length
            adjusted_scores = scores + vocab_adjustment[:, :scores.size(-1)].to(scores.device)
            
            if position > 0 and position < 3:
                security_boost = normalized_stats[:scores.size(-1)].to(scores.device)
                security_boost = torch.where(security_boost > 0, security_boost, 0)
                adjusted_scores += security_boost.unsqueeze(0)
            
            return adjusted_scores

        if 'logits_processor' not in kwargs:
            from transformers import LogitsProcessorList
            kwargs['logits_processor'] = LogitsProcessorList()
        kwargs['logits_processor'].append(security_logits_processor)
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
  
    def update_token_statistics(self, batch: Dict[str, torch.Tensor]):
        """Update token security statistics"""
        with torch.no_grad():
            vulnerable_tokens = batch['input_ids_vulnerable'].flatten()
            for token_id in vulnerable_tokens:
                if token_id != self.tokenizer.pad_token_id:
                    self.token_security_stats[token_id] -= 0.01
            
            secure_tokens = batch['input_ids_secure'].flatten()
            for token_id in secure_tokens:
                if token_id != self.tokenizer.pad_token_id:
                    self.token_security_stats[token_id] += 0.01
            
            self.token_security_stats.clamp_(-1, 1)
    
    def save_pretrained(self, save_directory: str):
        """Save model and components"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save config
        config = {
            'target_modules': self.target_modules,
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'lora_rank': self.lora_rank,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'num_hidden_layers_to_use': self.num_hidden_layers_to_use,
            'aggregation_method': self.aggregation_method,
        }
        
        with open(os.path.join(save_directory, 'security_config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save PEFT model
        self.model.save_pretrained(save_directory)
        
        # Save additional components
        checkpoint = {
            'security_analyzer': self.security_analyzer.state_dict(),
            'feature_aggregator': self.feature_aggregator.state_dict(),
            'token_security_stats': self.token_security_stats,
        }
        
        torch.save(checkpoint, os.path.join(save_directory, 'security_components.pt'))
    
    @classmethod
    def from_pretrained(cls, save_directory: str, base_model_name_or_path: str, 
                       tokenizer: AutoTokenizer, device: str = "cpu"):
        """Load from saved model"""
        # Load config
        with open(os.path.join(save_directory, 'security_config.json'), 'r') as f:
            config = json.load(f)
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            device_map='auto' if torch.cuda.is_available() else None
        )
        
        # Load PEFT model
        model = PeftModel.from_pretrained(base_model, save_directory)
        
        # Create wrapper instance
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        
        instance.model = model
        instance.tokenizer = tokenizer
        instance.device = device
        instance.vocab_size = config['vocab_size']
        instance.hidden_size = config['hidden_size']
        instance.lora_rank = config['lora_rank']
        instance.lora_alpha = config['lora_alpha']
        instance.lora_dropout = config['lora_dropout']
        instance.target_modules = config['target_modules']
        instance.num_hidden_layers_to_use = config.get('num_hidden_layers_to_use', 4)
        instance.aggregation_method = config.get('aggregation_method', 'weighted')
        instance.logger = logging.getLogger(__name__)
        
        # Load security components
        instance.security_analyzer = EnhancedSecurityAnalyzer(
            vocab_size=instance.vocab_size,
            hidden_size=instance.hidden_size,
            num_layers=instance.num_hidden_layers_to_use,
            embed_dim=128,
            aggregation_method=instance.aggregation_method
        ).to(device)
        
        instance.feature_aggregator = MultiLayerAggregator(
            num_layers=instance.num_hidden_layers_to_use,
            hidden_size=instance.hidden_size,
            aggregation_method=instance.aggregation_method
        ).to(device)
        
        # Load checkpoint
        checkpoint = torch.load(
            os.path.join(save_directory, 'security_components.pt'),
            map_location=device
        )
        
        instance.security_analyzer.load_state_dict(checkpoint['security_analyzer'])
        instance.feature_aggregator.load_state_dict(checkpoint['feature_aggregator'])
        instance.register_buffer('token_security_stats', checkpoint['token_security_stats'])
        
        return instance

class SecurityCodeDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length: int = 512, mode: str = 'train', 
                 logger: Optional[logging.Logger] = None):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
        self.logger = logger or logging.getLogger(__name__)
        self.data = []
        
        # Load data
        data_file = os.path.join(self.data_dir, self.mode, f'{self.mode}.jsonl')
        self.logger.info(f"Loading data from {data_file}")
        
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                
                # Encode
                vulnerable_enc = self._encode_code(item['func_src_before'])
                secure_enc = self._encode_code(item['func_src_after'])
                
                # Skip if too long
                if len(vulnerable_enc['input_ids']) > max_length or len(secure_enc['input_ids']) > max_length:
                    continue
                
                self.data.append({
                    'func_name': item['func_name'],
                    'input_ids_vulnerable': torch.tensor(vulnerable_enc['input_ids'], dtype=torch.long),
                    'attention_mask_vulnerable': torch.tensor(vulnerable_enc['attention_mask'], dtype=torch.long),
                    'input_ids_secure': torch.tensor(secure_enc['input_ids'], dtype=torch.long),
                    'attention_mask_secure': torch.tensor(secure_enc['attention_mask'], dtype=torch.long),
                })
        
        self.logger.info(f"Loaded {len(self.data)} examples from {mode} set")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def _encode_code(self, code: str) -> Dict[str, List[int]]:
        return self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True
        )


class SecurityTrainer:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        data_dir: str,
        cache_dir: str = './cache',
        logger: Optional[logging.Logger] = None,
        target_modules: Optional[List[str]] = None,
        learning_rate: float = 2e-4,
        batch_size: int = 32,
        max_epochs: int = 10,
        grad_acc_steps: int = 1,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        logging_steps: int = 10,
        eval_steps: int = 100,
        save_steps: int = 500,
        patience: int = 3,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        max_length: int = 512,
        use_kl_loss: bool = False,
        num_hidden_layers_to_use: int = 4,  # 新参数：使用的层数
        aggregation_method: str = 'weighted'  # 新参数：聚合方法
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Save parameters
        self.model_path = model_path
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        
        # Load tokenizer and model
        self.logger.info(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto' if torch.cuda.is_available() else None
        )

        # 冻结原始模型参数
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Enable gradient checkpointing
        if hasattr(base_model, 'gradient_checkpointing_enable'):
            base_model.gradient_checkpointing_enable()
        
        # Initialize security-aware model with PEFT
        self.logger.info("Initializing Security-Aware LoRA Model with Multi-Layer Support...")
        self.model = SecurityAwareLoRAModel(
            base_model,
            tokenizer=self.tokenizer,
            target_modules=target_modules,
            device=self.device,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_hidden_layers_to_use=num_hidden_layers_to_use,
            aggregation_method=aggregation_method,
            logger=self.logger
        ).to(self.device)
        
        # Set KL loss flag
        self.model.use_kl_loss = use_kl_loss
        
        # Load datasets
        self.logger.info(f"Loading datasets from {data_dir}")
        self.train_dataset = SecurityCodeDataset(
            data_dir, self.tokenizer, max_length=max_length, mode='train', logger=self.logger
        )
        self.val_dataset = SecurityCodeDataset(
            data_dir, self.tokenizer, max_length=max_length, mode='val', logger=self.logger
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Get trainable parameters
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                self.logger.debug(f"Trainable: {name}")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Scheduler
        steps_per_epoch = len(self.train_loader) // grad_acc_steps
        self.total_steps = steps_per_epoch * max_epochs
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(self.total_steps * warmup_ratio),
            num_training_steps=self.total_steps
        )
        
        # Training parameters
        self.cache_dir = cache_dir
        self.max_epochs = max_epochs
        self.grad_acc_steps = grad_acc_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.patience = patience
        self.patience_counter = 0
        self.best_val_loss = float('inf')
        
        # Log configuration
        total_params = sum(p.numel() for p in base_model.parameters())
        trainable_count = sum(p.numel() for p in trainable_params)
        
        self.logger.info('=' * 50)
        self.logger.info('Training Configuration')
        self.logger.info('=' * 50)
        self.logger.info(f'Model: {model_name}')
        self.logger.info(f'Total parameters: {total_params:,}')
        self.logger.info(f'Trainable parameters: {trainable_count:,}')
        self.logger.info(f'Trainable percentage: {100 * trainable_count / total_params:.2f}%')
        self.logger.info(f'Batch size: {batch_size}')
        self.logger.info(f'Learning rate: {learning_rate}')
        self.logger.info(f'Max epochs: {max_epochs}')
        self.logger.info(f'Use KL loss: {use_kl_loss}')
        self.logger.info(f'Hidden layers to use: {num_hidden_layers_to_use}')
        self.logger.info(f'Aggregation method: {aggregation_method}')
        self.logger.info('=' * 50)
    
    def train(self):
        """Main training loop"""
        global_step = 0
        best_checkpoint_path = None
        
        self.logger.info("Starting training...")
        
        for epoch in range(self.max_epochs):
            self.model.train()
            
            epoch_losses = defaultdict(float)
            batch_count = 0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.max_epochs}")
            
            for batch_idx, batch in enumerate(progress_bar):
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Update token statistics
                self.model.update_token_statistics(batch)
                
                # Compute loss
                loss, loss_dict = self.model.compute_loss(batch)
                
                # Accumulate losses
                for key, value in loss_dict.items():
                    epoch_losses[key] += value
                batch_count += 1
                
                # Gradient accumulation
                if self.grad_acc_steps > 1:
                    loss = loss / self.grad_acc_steps
                
                # Backward
                loss.backward()
                
                # Update weights
                if (batch_idx + 1) % self.grad_acc_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    global_step += 1
                    
                    # Update progress bar
                    avg_losses = {k: v / batch_count for k, v in epoch_losses.items()}
                    postfix_dict = {
                        'loss': f"{avg_losses['total']:.4f}",
                        'gen': f"{avg_losses['generation']:.4f}",
                        'sec': f"{avg_losses['security']:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                        'grad': f"{grad_norm:.2f}"
                    }
                    
                    if 'kl' in avg_losses:
                        postfix_dict['kl'] = f"{avg_losses['kl']:.4f}"
                    
                    progress_bar.set_postfix(postfix_dict)
                    
                    # Logging
                    if global_step % self.logging_steps == 0:
                        log_msg = f"Step {global_step}: Total={avg_losses['total']:.4f}, Gen={avg_losses['generation']:.4f}, Sec={avg_losses['security']:.4f}"
                    
                        if 'token_contrastive' in avg_losses:
                            log_msg += f", TokenCon={avg_losses['token_contrastive']:.4f}"
                        if 'feature_contrastive' in avg_losses:
                            log_msg += f", FeatureCon={avg_losses['feature_contrastive']:.4f}"
                        if 'kl' in avg_losses:
                            log_msg += f", KL={avg_losses['kl']:.6f}"
                            
                        log_msg += f", LR={self.scheduler.get_last_lr()[0]:.2e}, Grad={grad_norm:.2f}"
                        self.logger.info(log_msg)
                    
                    # Evaluation
                    if global_step % self.eval_steps == 0:
                        val_loss = self.evaluate()
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            best_checkpoint_path = self.save_checkpoint(epoch, val_loss, is_best=True)
                            self.logger.info(f"New best model! Val Loss: {val_loss:.4f}")
                        else:
                            self.patience_counter += 1
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(epoch, avg_losses['total'])
                
                # Clear cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            # End of epoch evaluation
            val_loss = self.evaluate()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_checkpoint_path = self.save_checkpoint(epoch, val_loss, is_best=True)
                self.logger.info(f"New best model saved! Val Loss: {val_loss:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            self.logger.info(
                f"Epoch {epoch+1} completed | Val Loss: {val_loss:.4f} | "
                f"Best: {self.best_val_loss:.4f} | Patience: {self.patience_counter}/{self.patience}"
            )
        
        self.logger.info(f"Training completed! Best checkpoint: {best_checkpoint_path}")
    
    def evaluate(self):
        """Evaluation function"""
        self.logger.info("Evaluating...")
        self.model.eval()
        
        total_losses = defaultdict(float)
        batch_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating'):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                loss, loss_dict = self.model.compute_loss(batch)
                
                for key, value in loss_dict.items():
                    total_losses[key] += value
                batch_count += 1
        
        # Average losses
        avg_losses = {k: v / batch_count for k, v in total_losses.items()}
        
        log_msg = f"Validation - Total: {avg_losses['total']:.4f}, Gen: {avg_losses['generation']:.4f}, Sec: {avg_losses['security']:.4f}"

        if 'kl' in avg_losses:
            log_msg += f", KL: {avg_losses['kl']:.6f}"
            
        self.logger.info(log_msg)
        
        return avg_losses['total']
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save checkpoint"""
        if is_best:
            save_dir = os.path.join(self.cache_dir, 'checkpoint-best')
        else:
            save_dir = os.path.join(self.cache_dir, f'checkpoint-epoch{epoch+1}')
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(save_dir)
        
        # Save training state
        training_state = {
            'epoch': epoch,
            'global_step': global_step if 'global_step' in locals() else 0,
            'best_val_loss': self.best_val_loss,
            'val_loss': val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'model_path': self.model_path,
        }
        
        torch.save(training_state, os.path.join(save_dir, 'training_state.pt'))
        
        self.logger.info(f'Checkpoint saved to {save_dir}')
        return save_dir


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(name: str, log_file: str, level=logging.INFO):
    """Get logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # File handler
    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger



def main():
    """Main training function"""

    import argparse
    
    parser = argparse.ArgumentParser(description='Security-Aware LoRA Training')
    parser.add_argument('--model_name', type=str, default='qwen2.5-7b', required=True)
    parser.add_argument('--aggregation_method', type=str, default='attention',
                       choices=['attention', 'last_layer', 'average'],
                       help='Aggregation method for multi-layer hidden states')
    
    args = parser.parse_args()

    MODEL_PATHS = {
        'deepseek-1.3b':'deepseek-ai/deepseek-coder-1.3b-base',
        'deepseek-6.7b':'deepseek-ai/deepseek-coder-6.7b-base',
        'qwen2.5-7b': 'Qwen/Qwen2.5-Coder-7B',
        'qwen2.5-3b': 'Qwen/Qwen2.5-Coder-3B',
        'qwen2.5-0.5b': 'Qwen/Qwen2.5-Coder-0.5B',
        'seedcoder-8b': 'ByteDance-Seed/Seed-Coder-8B-Base',
    }

    model_path = MODEL_PATHS[args.model_name]

    # Configuration
    config = {
        'model_name': '',
        'model_path': '',
        'data_dir': '../data_train_val',
        'cache_dir': '../trained',
        'target_modules': None,
        'learning_rate': 2e-5,
        'batch_size': 8,
        'max_epochs': 5,
        'grad_acc_steps': 2,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0,
        'logging_steps': 5,
        'eval_steps': 1000,
        'save_steps': 1000,
        'patience': 3,
        'lora_rank': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.1,
        'max_length': 1024,
        'use_kl_loss': True,
        'num_hidden_layers_to_use': 4, 
        'aggregation_method': '',
    }

    
    config['model_path'] = model_path
    config['aggregation_method'] = args.aggregation_method
    config['model_name'] = args.model_name + '-deepguard-' + args.aggregation_method

    # Create output directory
    config['cache_dir'] = os.path.join(config['cache_dir'], config['model_name'])
    os.makedirs(config['cache_dir'], exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(config['cache_dir'], 'train.log')
    logger = get_logger(__name__, log_file)
    config['logger'] = logger
    
    # Log configuration
    logger.info('Configuration:')
    for key, value in config.items():
        if key != 'logger':
            logger.info(f'  {key}: {value}')
    logger.info('=' * 80)
    
    # Set seed
    set_seed(42)
    
    # Create trainer and train
    trainer = SecurityTrainer(**config)
    trainer.train()
    
    logger.info('=' * 80)
    logger.info('Training completed successfully!')


if __name__ == "__main__":
    main()