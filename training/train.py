import numpy as np
import torch
import torch.nn as nn
import fire
import random

from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from core.data import SubgoalEvalDataset
from core.evaluator import SubgoalEvalNetwork


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _run_epoch(model, loader, optimizer, criterion, device, phase):
    is_train = phase == "train"
    if is_train:
        model = model.train()
    else:
        model = model.eval()

    cumulative_loss = 0.
    total_predictions = 0

    if is_train:
        optimizer.zero_grad()

    for X_batch, Y_batch in tqdm(loader):
        with torch.set_grad_enabled(is_train):
            pred_batch = model.forward(subgoal=X_batch["subgoal"].to(device),
                                       previous_robot_trajectory=X_batch["robot"].to(device),
                                       previous_pedestrians_trajectories=X_batch["peds"].to(device),
                                       predicted_pedestrians_trajectories=X_batch["peds_pred"].to(device),
                                       predicted_pedestrians_covs=X_batch["peds_covs"].to(device))
            loss = criterion.forward(pred_batch, Y_batch.to(device))

        if is_train:
            loss.backward()
            optimizer.step()

        n_predictions = pred_batch.shape[0]
        total_predictions = total_predictions + n_predictions
        cumulative_loss = cumulative_loss + loss.item() * n_predictions

    mean_loss = cumulative_loss / total_predictions
    return mean_loss


def main(model_name: str, data_root: str, epochs: int = 100, device: str = "cuda", output_dir: str = "."):
    data_root = Path(data_root)
    output_dir = Path(output_dir)

    dataset_train = SubgoalEvalDataset(model_name,
                                       str(data_root / "datasets" / f"trajectory_dataset_{model_name}.json"),
                                       str(data_root / "statistics" / f"stats_{model_name}.json"),
                                       trials=[i for i in range(0, 25)])
    dataset_val = SubgoalEvalDataset(model_name,
                                       str(data_root / "datasets" / f"trajectory_dataset_{model_name}.json"),
                                       str(data_root / "statistics" / f"stats_{model_name}.json"),
                                     trials=[i for i in range(25, 30)])

    loader_train = DataLoader(dataset_train, batch_size=1, shuffle=False)
    loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False)

    model = SubgoalEvalNetwork()
    model = model.to(device)
    model = model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.BCELoss()

    best_loss = np.inf

    for epoch in range(epochs):
        train_loss = _run_epoch(model=model,
                                loader=loader_train,
                                optimizer=optimizer,
                                criterion=criterion,
                                device=device,
                                phase="train")
        output_str = f"Epoch {epoch}: train {round(train_loss, 5)}"
        if epoch % 5 == 0:
            val_loss = _run_epoch(model=model,
                                    loader=loader_val,
                                    optimizer=optimizer,
                                    criterion=criterion,
                                    device=device,
                                    phase="val")
            output_str = output_str + f" val {round(val_loss, 5)}"

            if val_loss < best_loss:
                best_loss = val_loss
                model = model.to("cpu")
                output_dir.mkdir(exist_ok=True, parents=True)
                torch.save(model.state_dict(), str(output_dir / f"{model_name}.pth.tar"))
                model = model.to(device)

        print(output_str)


if __name__ == '__main__':
    fire.Fire(main)



    # for item in dataset:
    #     test = model.forward(subgoal=item["subgoal"].unsqueeze(0),
    #                          previous_robot_trajectory=item["robot"].unsqueeze(0),
    #                          previous_pedestrians_trajectories=item["peds"].unsqueeze(0),
    #                          predicted_pedestrians_trajectories=item["peds_pred"].unsqueeze(0),
    #                          predicted_pedestrians_covs=item["peds_covs"].unsqueeze(0))
