from __future__ import print_function

import argparse
import functools

import mlflow
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms

import modelling


def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100.0 * correct / len(pred)
            mlflow.log_metrics({"train_loss": loss.item(), "train_accuracy": accuracy})
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), accuracy, loss.item()
                )
            )


def test(args, model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / len(test_loader.dataset)
    mlflow.log_metrics({"test_loss": test_loss, "test_accuracy": test_accuracy})
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), test_accuracy
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, metavar="N", help="number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")

    parser.add_argument("--run-name", type=str, default=None, help="Mlflow run name.")

    parser.add_argument(
        "--save-model", action="store_true", default=False, help="For Saving the current Model"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dataset = functools.partial(
        datasets.MNIST,
        root="../data",
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset(train=True), batch_size=args.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset(train=False), batch_size=args.test_batch_size, shuffle=True
    )

    model = modelling.Net()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    mlflow.set_experiment("MNIST CNN")
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params(vars(args))
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, epoch)
            test(args, model, test_loader)
            scheduler.step()

        if args.save_model:
            model_path = "mnist_cnn.pt"
            torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
