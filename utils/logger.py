import os
import time
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, PartialState
import torch
from collections import deque
from utils.misc import is_distributed, is_main_process
import yaml
import json

state = PartialState()

class TPS:
    """
    Time Per Step.
    """
    def __init__(self, windows_size: int = 50):
        self.tps_deque = deque(maxlen=windows_size)     # time per step.

    def update(self, tps: float):
        self.tps_deque.append(tps)

    @property
    def average(self):
        tps_list = list(self.tps_deque)
        _average = sum(tps_list) / len(tps_list)
        if not is_distributed():
            return _average
        else:
            _average = torch.tensor(_average, dtype=torch.float32, device="cuda")
            torch.distributed.all_reduce(_average, op=torch.distributed.ReduceOp.AVG)
            # print(_average)
            return _average.item()

    def eta(self, total_steps: int, current_steps: int):
        return self.average * (total_steps - current_steps)

    @classmethod
    def timestamp(cls):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.time()

    @classmethod
    def format(cls, seconds: float):
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h)}:{int(m)}:{int(s)}"

class Logger:
    """Logger for training metrics"""
    
    def __init__(self, log_dir: str, accelerator: Accelerator, config: dict | None = None, use_tensorboard: bool = True, tensorboard_flush_secs: int = 30):
        """
        Initialize Logger
        
        Args:
            log_dir: Directory to save tensorboard logs
            accelerator: Accelerator instance for distributed training
            flush_secs: How often to flush the tensorboard writer
        """
        self.accelerator = accelerator
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        
        # Only create writer on main process
        if accelerator.is_main_process:
            os.makedirs(log_dir, exist_ok=True)
            if self.use_tensorboard:
                self.tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=tensorboard_flush_secs)
                self.accelerator.print(f"TensorBoard logging initialized at: {log_dir}")
            else:
                self.tb_writer = None
        else:
            self.tb_writer = None
    
    # config related 
    def config(self, config: dict):
        self._print_config(config=config)
        self._save_config(config=config, filename="config.yaml")
        return
    
    @state.on_main_process
    def _print_config(self, config: dict):
        print(self._colorize(log="[Runtime Config]", log_type="success"), end=" ")
        for _ in config:
            print(f"{_.lower()}: {config[_]} | ", end="")
        print("", end="\n")

    @state.on_main_process
    def _save_config(self, config: dict, filename: str = "config.yaml"):
        self._write_dict_to_yaml(x=config, filename=filename, mode="w")
        return
    
    def _write_dict_to_yaml(self, x: dict, filename: str, mode: str = "w"):
        with open(os.path.join(self.log_dir, filename), mode=mode) as f:
            yaml.dump(x, f, allow_unicode=True)
        return

    def _write_dict_to_json(self, log: dict, filename: str, mode: str = "w"):
        """
        Logger writes a dict log to a .json file.

        Args:
            log (dict): A dict log.
            filename (str): Log file's name.
            mode (str): File writing mode, "w" or "a".
        """
        with open(os.path.join(self.logdir, filename), mode=mode) as f:
            f.write(json.dumps(log, indent=4))
            f.write("\n")
        return
    
    @staticmethod
    def _is_to_do(only_main: bool = True):
        return is_main_process() or not only_main

    @staticmethod
    def _colorize(log: str, log_type: str):
        if log_type == "info":
            return f"\033[1;36m{log}\033[0m"
        elif log_type == "warning":
            return f"\033[1;33m{log}\033[0m"
        elif log_type == "error":
            return f"\033[1;31m{log}\033[0m"
        elif log_type == "success":
            return f"\033[1;32m{log}\033[0m"
        else:
            raise ValueError(f"Unknown log type: {log_type}.")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value"""
        if self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars with a common main tag"""
        if self.tb_writer is not None:
            self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_loss_dict(self, loss_dict: dict, step: int, prefix: str = "train"):
        """
        Log loss dictionary to tensorboard
        
        Args:
            loss_dict: Dictionary containing loss values
            step: Global training step
            prefix: Prefix for the log tags (e.g., 'train', 'val')
        """
        if self.tb_writer is not None:
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
        if self.tb_writer is not None:
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group['lr']
                self.log_scalar(f"lr/group_{i}", lr, step)
    
    def log_model_parameters(self, model, step: int):
        """Log model parameter statistics"""
        if self.tb_writer is not None:
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
        if self.tb_writer is not None:
            self.tb_writer.add_image(tag, img_tensor, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram of values"""
        if self.tb_writer is not None:
            self.tb_writer.add_histogram(tag, values, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text"""
        if self.tb_writer is not None:
            self.tb_writer.add_text(tag, text, step)
    
    def flush_tb_writer(self):
        """Flush the writer"""
        if self.tb_writer is not None:
            self.tb_writer.flush()
    
    def close_tb_writer(self):
        """Close the writer"""
        if self.tb_writer is not None:
            self.tb_writer.close()
    
    def __del__(self):
        """Destructor to ensure writer is closed"""
        self.close_tb_writer()
    
    # txt related
    def info(self, log: str, only_main: bool = True):
        self._print(log=f"{self._colorize(log='[INFO]', log_type='info')} {log}", only_main=only_main)
        self._save(log=f"[INFO] {log}", only_main=only_main)
        return

    def warning(self, log: str, only_main: bool = True):
        self._print(log=f"{self._colorize(log='[WARNING]', log_type='warning')} {log}", only_main=only_main)
        self._save(log=f"[WARNING] {log}", only_main=only_main)
        return

    def success(self, log: str, only_main: bool = True):
        self._print(log=f"{self._colorize(log='[SUCCESS]', log_type='success')} {log}", only_main=only_main)
        self._save(log=f"[SUCCESS] {log}", only_main=only_main)
        return
    
    def _print(self, log: str, only_main: bool = True):
        if self._is_to_do(only_main=only_main):
            print(log)

    def _save(self, log: str, filename: str = "log.txt", mode: str = "a", only_main: bool = True, end: str = "\n"):
        if self._is_to_do(only_main=only_main):
            with open(os.path.join(self.log_dir, filename), mode=mode) as f:
                f.write(log + end)
        return


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