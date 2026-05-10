import gc
from copy import deepcopy
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm


def train_the_model(
    model,
    train_dataloader,
    val_dataloader,
    model_path_save,
    device,
    num_epochs=100,
    learning_rate=5e-6,
    delta=0.6,
    temperature=np.log(0.07),
    patience=10,
    no_early_stopping=False,
    wandb_run=None,
    lr_step_size=10,
):
    del delta  # kept for signature parity with notebook
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def eval_epoch(model, dataloader, test_mode=False):
        eval_loss = 0
        model.eval()

        with torch.no_grad(), tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
            for _, (text_emb, audio_emb, image_emb) in pbar:
                text_emb = text_emb.to(device)
                audio_emb = audio_emb.to(device)
                image_emb = image_emb.to(device)

                final_emb = model(audio_emb, image_emb)

                final_emb = l2_normalize(final_emb)
                text_emb = l2_normalize(text_emb)

                sim_matrix = torch.matmul(final_emb, text_emb.t())
                temperature_tensor = torch.tensor(temperature).to(device)
                sim_matrix *= torch.exp(temperature_tensor)

                labels = torch.arange(sim_matrix.size(0)).to(device)
                loss = (
                    nn.CrossEntropyLoss()(sim_matrix, labels)
                    + nn.CrossEntropyLoss()(sim_matrix.t(), labels)
                ) / 2

                eval_loss += loss.item()
                description = "Validation" if not test_mode else "Test"
                pbar.set_description(f"{description} Loss: {loss.item():.4f}")
        return eval_loss / len(dataloader)

    def l2_normalize(x, dim=-1):
        return x / x.norm(2, dim=dim, keepdim=True)

    def train_epoch(model, optimizer, dataloader, temperature):
        train_loss = 0
        model.train()
        with tqdm(enumerate(dataloader), total=len(dataloader)) as pbar:
            for _, (text_emb, audio_emb, image_emb) in pbar:
                text_emb = text_emb.to(device)
                audio_emb = audio_emb.to(device)
                image_emb = image_emb.to(device)

                final_emb = model(audio_emb, image_emb)
                final_emb = l2_normalize(final_emb)
                text_emb = l2_normalize(text_emb)

                sim_matrix = torch.matmul(final_emb, text_emb.t())
                temperature_tensor = torch.tensor(temperature).to(device)
                sim_matrix *= torch.exp(temperature_tensor)

                labels = torch.arange(sim_matrix.size(0)).to(device)
                loss = (
                    nn.CrossEntropyLoss()(sim_matrix, labels)
                    + nn.CrossEntropyLoss()(sim_matrix.t(), labels)
                ) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.set_description(f"Train Loss: {loss.item():.4f}")

        return train_loss / len(dataloader)

    def train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        model_path_save,
        epochs,
        temperature,
        patience_inner,
        no_early_stopping_inner,
        wandb_run_inner,
    ):
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        counter = 0
        epoch_num = epochs
        best_model = None

        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=0.4)

        for epoch in range(epochs):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            start_time = time()

            train_loss = train_epoch(model, optimizer, train_dataloader, temperature)
            val_loss = eval_epoch(model, val_dataloader, temperature)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            end_time = time()
            epoch_duration = end_time - start_time
            print(f"Epoch {epoch + 1} finished in {epoch_duration:.2f}s")
            print(
                f"[Epoch {epoch + 1}]\t"
                f"Train Loss: {train_loss:.6f}\t"
                f"Validation Loss: {val_loss:.6f}"
            )

            if wandb_run_inner is not None:
                wandb_run_inner.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "epoch": epoch + 1,
                    "epoch_duration_s": epoch_duration,
                    "lr": scheduler.get_last_lr()[0],
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = deepcopy(model)
                counter = 0
            else:
                counter += 1

            if not no_early_stopping_inner and counter >= patience_inner:
                print(
                    f"Early stopping after {patience_inner} epochs without improvement in validation loss."
                )
                if wandb_run_inner is not None:
                    wandb_run_inner.log({"early_stopping_epoch": epoch + 1})
                with open("stopped_epoch.txt", "w", encoding="utf-8") as f:
                    print(f"Stopped at epoch: {epoch + 1}")
                    f.write(f"Stopped at epoch: {epoch + 1}\n")
                    torch.save(best_model, model_path_save)
                    epoch_num = epoch + 1
                break

            scheduler.step()

        return train_losses, val_losses, epoch_num, best_model

    def plot_losses(train_losses, val_losses, epoch_num, save_path):
        import matplotlib
        matplotlib.use("Agg")  # headless backend, no display needed
        ls_epoch = [_ + 1 for _ in range(epoch_num)]
        plt.figure()
        plt.plot(ls_epoch, train_losses, color="r", label="train")
        plt.plot(ls_epoch, val_losses, color="b", label="validation")
        plt.title("Loss plot")
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.legend()
        plot_path = Path(save_path).with_suffix(".loss.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Loss plot saved to {plot_path}")

    train_losses, val_losses, epoch_num, best_model = train(
        model,
        optimizer,
        train_dataloader,
        val_dataloader,
        model_path_save,
        num_epochs,
        temperature,
        patience,
        no_early_stopping,
        wandb_run,
    )
    plot_losses(train_losses, val_losses, epoch_num, model_path_save)
    return best_model

