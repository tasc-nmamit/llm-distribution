import os
import torch
import torch.distributed.rpc as rpc
from models.model import ModelConfig, FinalModel
from helpers.data import DataLoader
from helpers.loss import estimate_loss

config = ModelConfig.get_config("shakespeare-200k")
data_loader = DataLoader(
    "input.txt", config.device, config.block_size, config.batch_size
)


def save_model(model):
    save_path = f"weights/shakespeare-gpt-{config.max_iters}.pth"
    torch.save(model.state_dict(), save_path)


def run_master():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    rpc.init_rpc("master", rank=0, world_size=3)

    model = FinalModel(config).to(config.device)
    # if os.path.exists(f"weights/shakespeare-gpt-{config.max_iters}.pth"):
    #     model.load_state_dict(
    #         torch.load(f"weights/shakespeare-gpt-{config.max_iters}.pth")
    #     )

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    for i, j in model.state_dict().items():
        print(i, j.shape)

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(data_loader.decode(model.generate(context, max_new_tokens=2000)[0]))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print("\n\n<-----------------------Training started------------------------->\n\n")

    for iter in range(config.max_iters):
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(
                model, config.eval_iters, config.device, data_loader.get_batch
            )
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        xb, yb = data_loader.get_batch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(
        "\n\n<-----------------------Training completed------------------------->\n\n"
    )

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print(data_loader.decode(model.generate(context, max_new_tokens=2000)[0]))

    save_model(model)

    rpc.shutdown()


if __name__ == "__main__":
    run_master()
