import torch
import typer
from pathlib import Path
import matplotlib.pyplot as plt
from model import Model
from data import MyDataset
import wandb
from sklearn.metrics import RocCurveDisplay, accuracy_score, precision_score, recall_score, f1_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
def train(
    lr: float = typer.Option(1e-3, "--lr"),
    epochs: int = typer.Option(5, "--epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "--batch_size"),
) -> None:
    print("Training day and night for {epochs} epochs")
    print(f"lr: {lr}")
    print(f"batch size: {batch_size}")
    run = wandb.init(
        project="corrupt_mnist",
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs},
    )
    dataset = MyDataset("data")
    model = Model()
    # add rest of your training code here
    model = Model()
    model.to(DEVICE)
    train_set, _ = dataset.load_plantvillage()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    statistics = {"train_loss": [], "train_accuracy": []}
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            preds.append(y_pred.detach().cpu())
            targets.append(target.detach().cpu())
            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}, accuracy: {accuracy}")
                
                # add a plot of the input images (log a list, not a batched tensor)
                images = [
                    wandb.Image(
                        (single_img.detach().cpu().clamp(0, 1) * 255)
                        .to(torch.uint8),
                        caption=f"Input {idx}",
                    )
                    for idx, single_img in enumerate(img[:5])
                ]
                wandb.log({"input_images": images})
                
                # add a plot of histogram of the gradients
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None], 0)
                wandb.log({"gradients": wandb.Histogram(grads.detach().cpu().numpy())})
                
                statistics["train_loss"].append(loss.item())
                statistics["train_accuracy"].append(accuracy)
    print("Training complete")
    Path("models").mkdir(parents=True, exist_ok=True)

    # Concatenate stored predictions/targets
    preds_tensor = torch.cat(preds, 0)
    targets_tensor = torch.cat(targets, 0)

    # ROC curves per class
    fig_roc, ax_roc = plt.subplots()
    for class_id in range(10):
        one_hot = torch.zeros_like(targets_tensor)
        one_hot[targets_tensor == class_id] = 1
        RocCurveDisplay.from_predictions(
            one_hot,
            preds_tensor[:, class_id],
            name=f"ROC curve for {class_id}",
            plot_chance_level=(class_id == 2),
            ax=ax_roc,
        )
    wandb.log({"roc": wandb.Image(fig_roc)})
    plt.close(fig_roc)

    final_accuracy = accuracy_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu())
    final_precision = precision_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")
    final_recall = recall_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")
    final_f1 = f1_score(targets_tensor.cpu(), preds_tensor.argmax(dim=1).cpu(), average="weighted")

    # first we save the model to a file then log it as an artifact
    torch.save(model.state_dict(), "models/model.pth")
    artifact = wandb.Artifact(
        name="corrupt_mnist_model",
        type="model",
        description="A model trained to classify corrupt MNIST images",
        metadata={"accuracy": final_accuracy, "precision": final_precision, "recall": final_recall, "f1": final_f1},
    )
    artifact.add_file("models/model.pth")
    run.log_artifact(artifact)
if __name__ == "__main__":
    typer.run(train)