import torch


@torch.no_grad()
def estimate_loss(model, eval_iters, device, get_batch):
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
