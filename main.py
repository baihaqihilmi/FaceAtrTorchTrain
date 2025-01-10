import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from torchvision import datasets, transforms
from torch.utils.data import random_split
import argparse
from models.model import MyModel
from torch import optim
from torch import nn
from data.dataset import UTKDataset
import os.path as osp
##TODO Create a pipeline for both Age and Gender Classification
##1 .  Fine tuned model from VGG-Face in Torch 
##2. Conevert the model to ONNX
## Convert the model to OpenVINO
##2 . INference The Model 



def parse_args():
    parser = argparse.ArgumentParser(description='Age and Gender Classification Training')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training and validation')
    parser.add_argument('--model_path', type=str, default='model.pth', help='Path to save the trained model')
    ## Input Image
    parser.add_argument('--model_name', type=str, default='resnet50', help='Name of the model to use for training')
    parser.add_argument('--img_size', type=int, nargs=2, default=[224, 224], help='Input image size')
    return parser.parse_args()


def main():
    args = parse_args()

    # Hyperparameters
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size

    # Device configuration
    device = 'cpu'

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load dataset
    datasets = UTKDataset(root_dir=osp.join('data', 'archive', 'UTKFace') , transform= transform)
    train_size = int(0.8 * len(datasets))
    val_size = len(datasets) - train_size
    train_dataset, val_dataset = random_split(datasets, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, criterion, optimizer
    model = MyModel(args)  # Replace with your model
    model = model.to(device)
    criterion = {
        "Age": nn.MSELoss(),
        "Gender": nn.BCEWithLogitsLoss(),
        "Race": nn.CrossEntropyLoss(),
    }
    optimizers = {
        "Age": optim.SGD(model.parameters(), lr=learning_rate),  # Adam optimizer for some part of the model
        "Gender": optim.SGD(model.parameters(), lr=learning_rate),  # SGD optimizer for another part
        "Race": optim.SGD(model.parameters(), lr=learning_rate)  # SGD optimizer for another part
    }
    model = model.to(device)
    # Train and validate the model
    train(model, train_loader, val_loader, criterion, optimizers, num_epochs, device)

    # Save the model
    torch.save(model.state_dict(), args.model_path)

def train(model, train_loader, val_loader, criterion, optimizers, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        running_loss = {
            "Age": 0,
            "Gender": 0,
            "Race": 0
        }

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            for k,v in labels.items(): 
                labels[k] = v.to(device)

    

            for k, v in running_loss.items():
                optimizers[k].zero_grad()

            outputs = model(inputs)

            for k, v in running_loss.items():
                if k == "Age" or k == "Gender":
                    loss = criterion[k](outputs[k].squeeze(), labels[k])  # Assuming labels are dictionaries with keys 'Age', 'Gender', etc.
                else : 
                    loss = criterion[k](outputs[k], labels[k].long())  # Assuming labels are dictionaries with keys 'Age', 'Gender', etc.

                running_loss[k] += loss.item()
                optimizers[k].step()

        print(f"Epoch [{epoch+1}/{num_epochs}], ")
        print(f"Losses: ")
        for k, v in running_loss.items():
            print(f"{k}: {v / len(train_loader)}")

        # Validation phase
        validate(model, val_loader, criterion, device)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = {
        "Age": 0.0,
        "Gender": 0.0,
        "Race": 0.0
    }
    all_labels = {
        "Age": [],
        "Gender": [],
        "Race": []
    }
    all_outputs = {
        "Age": [],
        "Gender": [],
        "Race": []
    }

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            for k in val_loss.keys():
                loss = criterion[k](outputs[k], labels[k])
                val_loss[k] += loss.item()

                all_labels[k].append(labels[k].cpu())
                all_outputs[k].append(outputs[k].cpu())

    for k in val_loss.keys():
        print(f"Validation Loss ({k}): {val_loss[k] / len(val_loader)}")

    # Metrics computation for each task
    if 'Gender' in all_outputs:
        gender_labels = torch.cat(all_labels['Gender'])
        gender_outputs = torch.cat(all_outputs['Gender'])
        gender_accuracy = accuracy_score(gender_labels, gender_outputs.argmax(dim=1))
        gender_precision = precision_score(gender_labels, gender_outputs.argmax(dim=1), average='binary')
        gender_recall = recall_score(gender_labels, gender_outputs.argmax(dim=1), average='binary')
        gender_f1 = f1_score(gender_labels, gender_outputs.argmax(dim=1), average='binary')

        print(f"Gender - Accuracy: {gender_accuracy*100}%, Precision: {gender_precision}, Recall: {gender_recall}, F1: {gender_f1}")

    if 'Age' in all_outputs:
        age_labels = torch.cat(all_labels['Age'])
        age_outputs = torch.cat(all_outputs['Age'])
        age_mae = mean_absolute_error(age_labels, age_outputs)
        age_mse = mean_squared_error(age_labels, age_outputs)

        print(f"Age - MAE: {age_mae}, MSE: {age_mse}")

    if 'Race' in all_outputs:
        race_labels = torch.cat(all_labels['Race'])
        race_outputs = torch.cat(all_outputs['Race'])
        race_accuracy = accuracy_score(race_labels, race_outputs.argmax(dim=1))
        race_precision = precision_score(race_labels, race_outputs.argmax(dim=1), average='weighted')
        race_recall = recall_score(race_labels, race_outputs.argmax(dim=1), average='weighted')
        race_f1 = f1_score(race_labels, race_outputs.argmax(dim=1), average='weighted')

        print(f"Race - Accuracy: {race_accuracy*100}%, Precision: {race_precision}, Recall: {race_recall}, F1: {race_f1}")



if __name__ == "__main__":
    main()
