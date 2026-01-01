import os
import sys
import time
from torch.profiler import profile, ProfilerActivity, record_function
import torch

benchmark_dataset = [
    {
        "prompt": "Tell me a joke",
        "max_new_tokens": 2000,
        "expected_output": '''<think>
Okay, the user asked for a joke. I need to come up with a fun and engaging one. Let me think of some common jokes that are easy to understand. Maybe something with animals or everyday situations.

Hmm, a classic joke about a cat and a mouse. The mouse is trying to escape, but the cat catches it. That's a classic setup. But maybe I can make it more unique. What if the mouse is trying to escape a trap? Or maybe the cat is trying to catch something else?

Wait, the user might prefer something that's not too obvious. Let me try a twist. What if the mouse is trying to escape a fire? The cat catches it. That's a good setup. But maybe the joke is too simple. Let me check if there's a better way.

Alternatively, a joke about a dog and a cat. The dog is chasing the cat, but the cat is running away. That's a common joke. But maybe I can add a twist. Like the dog is trying to catch something else. Or maybe the cat is trying to catch a mouse. 

Wait, the original idea with the mouse and the cat is simple. Let me make sure it's not too obvious. Maybe the mouse is trying to escape a trap, and the cat catches it. That's a classic setup. I think that's a solid joke. Let me put it together and make sure it's clear and funny.
</think>

Here's a classic and clever joke:  

**"Why did the mouse try to escape the cat?**  
Because it was caught in the act of escaping!"**  

😄<|im_end|>''',
    }
]


if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    assert torch.cuda.is_available(), "CUDA is not available"
    
    from nano.engine import Engine, EngineConfig
    engine = Engine(
        config=EngineConfig(
            model_path="Qwen/Qwen3-0.6B",
        )
    )
    
    data = benchmark_dataset[0]
    
    with torch.no_grad():
        total_time = 0
        batch_size = 2
        prompts = [data["prompt"]] * batch_size
        with record_function(f"inference_step"):
            print(f">>>>> Running query ... <<<<<")
            start_time = time.perf_counter()
            outputs = engine.generate_batch(prompts)
            assert len(outputs) == batch_size
            torch.cuda.synchronize()
            duration = time.perf_counter() - start_time
            total_time += duration
            print(f"Time taken: {duration:.2f} seconds")
            for i in range(len(outputs)):
                print(f'Output: {outputs[i]}')
                assert outputs[i].strip("\n") == data["expected_output"].strip("\n"), "Output mismatch"
        print(f"Time taken: {total_time:.2f} seconds")