import torch
from MUL_main_Infer import DiscourseParser

print(torch.cuda.device_count())
dp = DiscourseParser(device="cuda:5", batch_size=512)

results = dp.parse(
    [
        "I'm trying to run inference on a MMSR model. The system has two 2080Ti GPUs and I'm running PyTorch 1.1.0 on Ubuntu 18.04 with CUDA 10."
    ]
    * 1000
)

print(results)
