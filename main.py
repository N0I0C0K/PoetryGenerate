import torch
import time

from torch import Tensor
from rich.progress import Progress
from torch.utils.data import DataLoader, random_split
from dataset import PoetryData
from model import PoetryNet

batch_size = 48
lr = 0.001


class PoetryGen:
    def __init__(self) -> None:
        # self.device = (
        #     torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        # )
        self.device = torch.device("cuda:0")
        self.dataset = PoetryData(self.device, max_lines=12000)
        self.vocab_size = self.dataset.vocab_size
        train_data, test_data = random_split(
            self.dataset, [len(self.dataset) - 1000, 1000]
        )
        self.train_dataloader = DataLoader(train_data, batch_size, True)
        self.test_dataloader = DataLoader(test_data, batch_size, True)

        self.net = PoetryNet(self.vocab_size, self.device, embed_size=128).to(
            self.device
        )
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr)
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=2)
        self.loaded_checkpoint_file = None
        self.epoch = 0

        self.progress = Progress()

        import glob

        files = glob.glob("checkpoint-*.pth")
        for i, file in enumerate(files):
            print(f"{i}> {file}")
        if files:
            t = input(
                "choose check point to load, default is the last one, n to unload>"
            )
            if t != "n":
                self.load_checkpoint(files[int(t)])

    def save_checkpoint(self):
        file_name = (
            self.loaded_checkpoint_file
            or f'checkpoint-{time.strftime("%y%m%d-%H%M")}.pth'
        )
        with open(file_name, "wb") as file:
            torch.save(
                {
                    "net_state": self.net.state_dict(),
                    "optimizer_state": self.optimizer.state_dict(),
                    "epoch": self.epoch,
                },
                file,
            )
        print(f"save check point to {file_name}")
        self.loaded_checkpoint_file = file_name

    def load_checkpoint(self, file: str):
        ckpt = torch.load(file)
        self.net.load_state_dict(ckpt["net_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.epoch = ckpt["epoch"]

        self.loaded_checkpoint_file = file
        print(f"loaded check point: {file}")

    def generate(self):
        self.net.eval()
        src = torch.LongTensor([[0]]).to(self.device)
        tgt = torch.LongTensor([[0]]).to(self.device)

        self.net.encode(src)

    def train_epoch(self):
        self.net.train()

        # ignore <pad> which index is 2
        loss_f = self.loss_f  # torch.nn.CrossEntropyLoss(ignore_index=2)

        def get_padding_mask(tokens: Tensor) -> Tensor:
            return (tokens == 2).to(self.device)

        voacb_size = self.dataset.vocab_size
        for i, (src, tgt) in enumerate(self.train_dataloader):
            src, tgt = src.to(self.device), tgt.to(self.device)
            src_mask = get_padding_mask(src)

            dec_tgt = tgt[:, :-1]
            dec_tgt_mask = get_padding_mask(dec_tgt)

            tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                dec_tgt.size(1), self.device
            )

            out = self.net.forward(src, dec_tgt, tgt_mask, src_mask, dec_tgt_mask)
            # out = torch.max(out, 2).indices
            loss = loss_f.forward(out.reshape(-1, voacb_size), tgt[:, 1:].flatten())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 40 == 0:
                print(f"epoch={self.epoch} loss={loss.item():.4f} {i}")

    def evaluation(self):
        self.net.eval()

        loss_f = self.loss_f

        def get_padding_mask(tokens: Tensor) -> Tensor:
            return (tokens == 2).to(self.device)

        voacb_size = self.dataset.vocab_size

        loss_a = 0
        with torch.no_grad():
            for i, (src, tgt) in enumerate(self.test_dataloader):
                src, tgt = src.to(self.device), tgt.to(self.device)
                src_mask = get_padding_mask(src)

                dec_tgt = tgt[:, :-1]
                dec_tgt_mask = get_padding_mask(dec_tgt)

                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    dec_tgt.size(1), self.device
                )

                out = self.net.forward(src, dec_tgt, tgt_mask, src_mask, dec_tgt_mask)
                # out = torch.max(out, 2).indices
                loss = loss_f.forward(out.reshape(-1, voacb_size), tgt[:, 1:].flatten())
                loss_a += loss.item()

        print(
            f"Validation: epoch={self.epoch} avg loss={loss_a/len(self.test_dataloader):.4f}"
        )

    def training(self, train_epoch_nums: int = 36):
        self.progress.start()
        for i in range(train_epoch_nums):
            self.train_epoch()
            self.evaluation()
            self.epoch += 1

            self.save_checkpoint()


def main():
    model = PoetryGen()
    model.training()


if __name__ == "__main__":
    main()
