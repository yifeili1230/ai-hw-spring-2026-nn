import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

# ================== Simple CNN Model ==================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ================== Attacks ==================
def fgsm_attack(model, images, labels, epsilon):
    images.requires_grad = True
    outputs = model(images)
    loss = nn.CrossEntropyLoss()(outputs, labels)
    model.zero_grad()
    loss.backward()
    sign_data_grad = images.grad.sign()
    perturbed_images = images + epsilon * sign_data_grad
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    return perturbed_images

def pgd_attack(model, images, labels, epsilon=0.3, alpha=0.01, steps=40):
    original_images = images.clone().detach()
    adv_images = images.clone().detach().requires_grad_(True)
    
    for _ in range(steps):
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        adv_images = adv_images + alpha * adv_images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(original_images + eta, 0, 1).detach().requires_grad_(True)
    
    return adv_images

def momentum_ifgsm(model, images, labels, epsilon=0.3, alpha=0.01, steps=40, mu=0.9):
    original_images = images.clone().detach()
    adv_images = images.clone().detach().requires_grad_(True)
    g = torch.zeros_like(images)
    
    for _ in range(steps):
        outputs = model(adv_images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        
        g = mu * g + adv_images.grad / torch.norm(adv_images.grad, p=1)
        adv_images = adv_images + alpha * g.sign()
        eta = torch.clamp(adv_images - original_images, min=-epsilon, max=epsilon)
        adv_images = torch.clamp(original_images + eta, 0, 1).detach().requires_grad_(True)
    
    return adv_images

# ================== Evaluation ==================
def evaluate(model, dataloader, attack=None, epsilon=0.3):
    model.eval()
    correct = 0
    total = 0
    device = next(model.parameters()).device
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        
        if attack:
            if attack == 'fgsm':
                images = fgsm_attack(model, images, labels, epsilon)
            elif attack == 'pgd':
                images = pgd_attack(model, images, labels, epsilon)
            elif attack == 'momentum':
                images = momentum_ifgsm(model, images, labels, epsilon)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# ================== Main ==================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=False)

    # Model & Training
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training model...")
    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(trainloader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved as mnist_model.pth")

    # Evaluation
    print("\n" + "="*60)
    clean_acc = evaluate(model, testloader)
    print(f"Clean Recognition Rate: {clean_acc:.2f}%")

    epsilon = 0.3
    attacks = ['fgsm', 'pgd', 'momentum']
    
    results = {}
    for atk_name in attacks:
        asr = 100 - evaluate(model, testloader, attack=atk_name, epsilon=epsilon)
        results[atk_name] = asr
        print(f"{atk_name.upper()} Attack Success Rate (ASR): {asr:.2f}%")

    # Save results
    with open("results.txt", "w") as f:
        f.write(f"Clean Accuracy: {clean_acc:.2f}%\n")
        for atk, asr in results.items():
            f.write(f"{atk.upper()} ASR: {asr:.2f}%\n")
    print("Results saved to results.txt")