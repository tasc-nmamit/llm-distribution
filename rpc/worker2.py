import os
import torch.distributed.rpc as rpc
from models.model import Head, MultiHeadAttention, Block, FeedForward


def run_worker():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    rpc.init_rpc("worker2", rank=2, world_size=3)
    print("Worker initialized and ready.")
    rpc.shutdown()


if __name__ == "__main__":
    run_worker()
