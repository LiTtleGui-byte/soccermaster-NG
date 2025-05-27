import os
import time
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import torch


class TensorBoardLogger:
    """TensorBoard Logger for training metrics"""
    
    def __init__(self, log_dir: str, accelerator: Accelerator, flush_secs: int = 30):
        """
        Initialize TensorBoard Logger
        
        Args:
            log_dir: Directory to save tensorboard logs
            accelerator: Accelerator instance for distributed training
            flush_secs: How often to flush the tensorboard writer
        """
        self.accelerator = accelerator
        self.log_dir = log_dir
        
        # Only create writer on main process
        if accelerator.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=log_dir, flush_secs=flush_secs)
            self.accelerator.print(f"TensorBoard logging initialized at: {log_dir}")
        else:
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars with a common main tag"""
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_loss_dict(self, loss_dict: dict, step: int, prefix: str = "train"):
        """
        Log loss dictionary to tensorboard
        
        Args:
            loss_dict: Dictionary containing loss values
            step: Global training step
            prefix: Prefix for the log tags (e.g., 'train', 'val')
        """
        if self.writer is not None:
            # Log total loss
            total_loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
            self.log_scalar(f"{prefix}/total_loss", total_loss.item() if torch.is_tensor(total_loss) else total_loss, step)
            
            # Log individual losses
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    value = value.item()
                self.log_scalar(f"{prefix}/{key}", value, step)
    
    def log_learning_rate(self, optimizer, step: int):
        """Log learning rate"""
        if self.writer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                self.log_scalar(f"lr/group_{i}", lr, step)
    
    def log_model_parameters(self, model, step: int):
        """Log model parameter statistics"""
        if self.writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Log parameter statistics
                    self.log_scalar(f"params/{name}_mean", param.data.mean().item(), step)
                    self.log_scalar(f"params/{name}_std", param.data.std().item(), step)
                    self.log_scalar(f"params/{name}_norm", param.data.norm().item(), step)
                    
                    # Log gradient statistics
                    self.log_scalar(f"grads/{name}_mean", param.grad.data.mean().item(), step)
                    self.log_scalar(f"grads/{name}_std", param.grad.data.std().item(), step)
                    self.log_scalar(f"grads/{name}_norm", param.grad.data.norm().item(), step)
    
    def log_image(self, tag: str, img_tensor, step: int):
        """Log an image"""
        if self.writer is not None:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values"""
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.writer is not None:
            self.writer.add_text(tag, text, step)
    
    def flush(self):
        """Flush the writer"""
        if self.writer is not None:
            self.writer.flush()
    
    def close(self):
        """Close the writer"""
        if self.writer is not None:
            self.writer.close()
    
    def __del__(self):
        """Destructor to ensure writer is closed"""
        self.close()


class MetricsTracker:
    """Helper class to track and accumulate metrics"""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict: dict):
        """Update metrics with new values"""
        for key, value in metrics_dict.items():
            if torch.is_tensor(value):
                value = value.item()
            
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] += value
            self.counts[key] += 1
    
    def get_averages(self):
        """Get average values of all tracked metrics"""
        averages = {}
        for key in self.metrics:
            averages[key] = self.metrics[key] / self.counts[key]
        return averages
    
    def reset(self):
        """Reset all tracked metrics"""
        self.metrics.clear()
        self.counts.clear() 