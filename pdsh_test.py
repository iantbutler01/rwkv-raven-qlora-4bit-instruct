import os
import torch
import deepspeed
import accelerate

with open('sample_file.txt', 'w') as f:
        f.write("using pdsh for distributed setup is a success!")

