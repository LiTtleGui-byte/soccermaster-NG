import torch
import time
import threading
import numpy as np

MEMORY_FRACTION = 0.8
COMPUTE_INTENSITY = 2048 * 5
SLEEP_TIME = 0.1

def gpu_worker(device_id):
    device = torch.device(f'cuda:{device_id}')
    torch.cuda.set_device(device)
    
    try:
        total_mem = torch.cuda.get_device_properties(device_id).total_memory
        alloc_bytes = int(total_mem * MEMORY_FRACTION)
        
        print(f"GPU [{device_id}] Allocating {alloc_bytes/1024**3:.2f} GB and starting computation...")
        
        big_tensor = torch.empty((alloc_bytes // 4,), dtype=torch.float32, device=device)
        
        size = COMPUTE_INTENSITY
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        
        while True:
            c = torch.matmul(a, b)
            d = torch.relu(c)
            e = torch.nn.functional.normalize(d)
            a.add_(0.01 * torch.randn_like(a))
            
            time.sleep(SLEEP_TIME)
            
    except Exception as e:
        print(f"Error on GPU {device_id}: {str(e)}")

if __name__ == '__main__':
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs, starting computation-intensive tasks...")
    
    threads = []
    for gpu_id in range(num_gpus):
        thread = threading.Thread(target=gpu_worker, args=(gpu_id,))
        thread.daemon = True
        thread.start()
        threads.append(thread)
    
    try:
        while True:
            # for i in range(num_gpus):
            #     util = torch.cuda.utilization(i)
            #     mem_used = torch.cuda.memory_allocated(i)
            #     mem_total = torch.cuda.get_device_properties(i).total_memory
            #     print(f"GPU {i}: Util={util}%, Mem={mem_used/1024**3:.2f}/{mem_total/1024**3:.2f} GB")
            # time.sleep(10)
            pass
            
    except KeyboardInterrupt:
        print("\nStopping computation and releasing resources...")