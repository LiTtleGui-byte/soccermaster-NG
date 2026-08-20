import os
import time
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator, PartialState
import torch
from collections import deque, defaultdict
from utils.misc import is_distributed, is_main_process, distributed_world_size
import yaml
import json
from collections import deque, defaultdict

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

class Value:
    def __init__(self, window_size: int = 50):
        self.value_deque = deque(maxlen=window_size)
        self.total_value = 0.0
        self.total_count = 0

        self.value_sync: None | torch.Tensor = None
        self.total_value_sync = None
        self.total_count_sync = None

    def update(self, value):
        self.value_deque.append(value)
        self.total_value += value
        self.total_count += 1

    def sync(self):
        if is_distributed():
            torch.distributed.barrier()
            value_gather = [None] * distributed_world_size()
            total_gather = [None] * distributed_world_size()
            torch.distributed.all_gather_object(value_gather, list(self.value_deque))
            torch.distributed.all_gather_object(total_gather, [self.total_value, self.total_count])
            values = [v for v_list in value_gather for v in v_list]
            self.value_sync = torch.as_tensor(values)
            self.total_value_sync = sum([_[0] for _ in total_gather])
            self.total_count_sync = sum([_[1] for _ in total_gather])
        else:                   # only one gpu.
            self.value_sync = torch.as_tensor(list(self.value_deque))
            self.total_value_sync = self.total_value
            self.total_count_sync = self.total_count
        return

    def clear(self):
        self.value_deque.clear()
        self.total_value = 0.0
        self.total_count = 0

        self.value_sync: None | torch.Tensor = None
        self.total_value_sync = None
        self.total_count_sync = None

    def _check_sync(self):
        if self.value_sync is None:
            raise RuntimeError(f"Be sure to use .sync() before metric statistic.")
        return

    @property
    def average(self):
        self._check_sync()
        return self.value_sync.mean().item()

    @property
    def global_average(self):
        self._check_sync()
        return self.total_value_sync / self.total_count_sync

    @property
    def median(self):
        self._check_sync()
        return self.value_sync.median().item()

    def fmt(self, fmt):
        return fmt.format(
            median=self.median,
            average=self.average,
            global_average=self.global_average
        ) 
class Metrics:
    def __init__(self):
        self.metrics = defaultdict(Value)

    def update(self, name: str, value: float):
        if isinstance(value, torch.Tensor):
            value = value.item()
        self.metrics[name].update(value)
        return

    def sync(self):
        for name, value in self.metrics.items():
            value.sync()
        return

    def __getitem__(self, item):
        return self.metrics[item]

    # Not suitable for PyCharm debugging process, carefully use:
    # def __getattr__(self, item):
    #     return self.metrics[item]

    def __str__(self):
        s = str()
        for name, value in self.metrics.items():
            s += f"{name} = {value.average:.4f} ({value.global_average:.4f}); "
        return s

    def fmt(self, fmt):
        s = str()
        for name, value in self.metrics.items():
            s += f"{name} = {value.fmt(fmt=fmt)}; "
        return s

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
        self.tensorboard_flush_secs = tensorboard_flush_secs
        
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
    
    def log_loss_dict(self, loss_dict: dict, step: int, prefix: str = "train", count_sum: bool = True):
        """
        Log loss dictionary to tensorboard
        
        Args:
            loss_dict: Dictionary containing loss values
            step: Global training step
            prefix: Prefix for the log tags (e.g., 'train', 'val')
        """
        if self.tb_writer is not None:
            # Log total loss
            if count_sum:
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
                # Use group name if available, otherwise use index
                group_name = param_group['name']
                self.log_scalar(f"lr/{group_name}", lr, step)
    
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
            
    def mark_resume(self, resumed_epoch: int, global_step: int):
        """
        Mark resume point in tensorboard logs
        
        Args:
            resumed_epoch: The epoch from which training is resumed
            global_step: Current global step when resuming
        """
        if self.tb_writer is not None:
            # Add text log to indicate resume point
            resume_text = f"Training resumed from epoch {resumed_epoch + 1}, global_step {global_step}"
            self.tb_writer.add_text("training/resume_info", resume_text, global_step)
            
            # Add a marker scalar to indicate resume point
            self.tb_writer.add_scalar("training/resume_marker", 1.0, global_step)
            
            # Add current timestamp as additional information
            import time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            self.tb_writer.add_text("training/resume_timestamp", 
                                   f"Resumed at: {timestamp}", global_step)
            
            # Flush immediately to ensure the resume marker is recorded
            self.tb_writer.flush()
            
            # Also log to console and file
            self.info(f"TensorBoard: Marked resume point at epoch {resumed_epoch + 1}, global_step {global_step}")
    
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
    
    # about metrics
    def metrics(
            self,
            log: str,
            metrics,
            fmt: None | str = "{average:.4f} ({global_average:.4f})",
            statistic: None | str = "average",
            global_step: int = 0,
            prefix: None | str = None,
            x_axis_step: None | int = None,
            x_axis_name: None | str = None,
            filename: str = "log.txt",
            file_mode: str = "a",
            only_main: bool = True
    ):
        self.print_metrics(
            metrics=metrics,
            prompt=f"{self._colorize(log='[Metrics]', log_type='info')} {log}",
            fmt=fmt,
            only_main=only_main,
        )
        self.save_metrics(
            metrics=metrics,
            prompt=f"[Metrics] {log}",
            fmt=fmt,
            statistic=statistic,
            global_step=global_step,
            prefix=prefix,
            x_axis_step=x_axis_step,
            x_axis_name=x_axis_name,
            filename=filename,
            file_mode=file_mode,
            only_main=only_main,
        )
        
    def print_metrics(
            self, metrics: Metrics, prompt: str = "",
            fmt: str = "{average:.4f} ({global_average:.4f})",
            only_main: bool = True,
    ):
        if self._is_to_do(only_main):
            print(prompt, end="")
            print(metrics.fmt(fmt=fmt))
        return

    def save_metrics_to_file(self, metrics: Metrics, prompt: str = "",
                             fmt: str = "{average:.4f} ({global_average:.4f})",
                             filename: str = "log.txt", mode: str = "a", only_main: bool = True):
        if self._is_to_do(only_main):
            log = f"{prompt}{metrics.fmt(fmt=fmt)}"
            self._save(log=log, filename=filename, mode=mode)
        return

    def save_metrics(
            self,
            metrics: Metrics,
            prompt: str = "",
            fmt: None | str = "{average:.4f} ({global_average:.4f})",
            statistic: None | str = "average",
            global_step: int = 0,
            prefix: None | str = None,
            x_axis_step: None | int = None,
            x_axis_name: None | str = None,
            filename: str = "log.txt",
            file_mode: str = "a",
            only_main: bool = True,
    ):
        """
        Save the metrics into .txt/wandb.

        Args:
            metrics: The metrics to save.
            prompt: Prompt of logging metrics.
            fmt: Format for Metric Value. If fmt is None, we will not output these into the log.txt file.
            statistic: Which statistic is output to wandb.
                       If is None, we will not output these to wandb.
            global_step: Global step of metrics records, generally, we set it to total iter of model training.
            prefix: Prefix of all metrics.
                    If your metric name is "loss", prefix is "epoch", the final name of this metric is "epoch_loss".
            x_axis_step: A different X-axis value from global step.
            x_axis_name: Name of X-axis.
            filename: The filename for saving metrics log.
            file_mode: The file mode for saving, like "w" or "a".
            only_main: Only save the log in the main process.
        Returns:

        """
        if fmt is not None:
            self.save_metrics_to_file(
                metrics=metrics, prompt=prompt, fmt=fmt, filename=filename, mode=file_mode, only_main=only_main
            )
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
       
