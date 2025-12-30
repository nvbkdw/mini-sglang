import os
import sys
import time
from torch.profiler import profile, ProfilerActivity, record_function
import torch

benchmark_dataset = [
    {
        "prompt": "Tell me a joke",
        "max_new_tokens": 2000,
        "output_file": "joke.txt"
    }
]


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    assert torch.cuda.is_available(), "CUDA is not available"
    
    from nano.engine import Engine
    engine = Engine("Qwen/Qwen3-0.6B")
    
    data = benchmark_dataset[0]
    
    with torch.no_grad():
        total_time = 0
        prompts = [data["prompt"]] * 5
        with record_function(f"inference_step"):
            print(f">>>>> Running query ... <<<<<")
            start_time = time.perf_counter()
            content = engine.generate_batch(prompts)
            torch.cuda.synchronize()
            duration = time.perf_counter() - start_time
            total_time += duration
            print(f'Output: {content}')
            print(f"Time taken: {duration:.2f} seconds")
            for i in range(len(content)):
                with open(data["output_file"], "r") as f:
                    expected_content = f.read()
                    assert content[i] == expected_content[i], "Output mismatch"
        print(f"Time taken: {total_time:.2f} seconds")