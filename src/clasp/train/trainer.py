import gc
from copy import deepcopy
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
        return eval_loss

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

        return train_loss

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
    ):
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        counter = 0
        epoch_num = epochs
        best_model = None

        scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.4)

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
            print(f"Epoch {epoch + 1} finished in {end_time - start_time:.2f}s")
            print(
                f"[Epoch {epoch + 1}]\t"
                f"Train Loss: {train_loss:.6f}\t"
                f"Validation Loss: {val_loss:.6f}"
            )

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
                with open("stopped_epoch.txt", "w", encoding="utf-8") as f:
                    print(f"Stopped at epoch: {epoch + 1}")
                    f.write(f"Stopped at epoch: {epoch + 1}\n")
                    torch.save(best_model, model_path_save)
                    epoch_num = epoch + 1
                break

            scheduler.step()

        return train_losses, val_losses, epoch_num, best_model

    def plot_loss(loss, epoch_num, label):
        ls_epoch = [_ + 1 for _ in range(epoch_num)]
        plt.plot(ls_epoch, loss, color="r", label=label)
        plt.title("Loss plot")
        plt.ylabel("Loss")
        plt.xlabel("epoch")
        plt.legend()
        plt.show()

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
    )
    plot_loss(train_losses, epoch_num, "train")
    plot_loss(val_losses, epoch_num, "validation")
    return best_model

