import torch


class DataLoader:
    def __init__(self, input_file, device, block_size, batch_size):
        self.device = device
        self.block_size = block_size
        self.batch_size = batch_size

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))

        # Create mappings
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Prepare data
        data = self.encode(text)
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def encode(self, s):
        return torch.tensor([self.stoi[c] for c in s], device=self.device)

    def decode(self, l):
        return "".join([self.itos[i] for i in l.cpu().tolist()])

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(
            len(data) - self.block_size, (self.batch_size,), device=self.device
        )
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x, y
