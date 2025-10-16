import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.models import BetaVAE

# Training the Disentangled Beta-VAE
def train_beta_vae(betavae, dataloader, beta, optimizer, num_epochs, device):
    betavae.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in enumerate(dataloader, 0):
            real_cpu = data[0].to(device)
            optimizer.zero_grad()

            recon_real, mu, logvar, latent_real = betavae(real_cpu)
            loss = betavae.loss_function(recon_real, real_cpu, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_values.append(total_loss/len(dataloader.dataset))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataloader.dataset):.4f}")

# Main script
if __name__ == '__main__':
    # Parameters
    latent_dim = 9
    beta = 2.0
    image_size = 64
    batch_size = 12
    num_epochs = 100
    lr_vae = 1e-4
    loss_values = []
    beta1 = 0.5
    beta2 = 0.999
    nc = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data preparation (Adjust the path as needed)
    img_path = '/dgxb_home/se21pphy004/Multiclass_Metasurface/Uni_data/Images/'
    
    dataset = dset.ImageFolder(root=img_path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor()
                               ]))
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Disentangled Beta-VAE
    betavae = BetaVAE(n_qubits = latent_dim, nc = nc, beta = beta).to(device)
    optimizer = optim.Adam(betavae.parameters(), lr=lr_vae, betas=(beta1, beta2))

    # Train the Disentangled Beta-VAE
    train_beta_vae(betavae, dataloader, beta, optimizer, num_epochs, device)

    # Save the pretrained Disentangled Beta-VAE model
    torch.save(betavae.state_dict(), 'pretrained_beta_vae_uni.pth')

    # Plot the loss values
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, color='b', marker='s', markevery=10, label='pretrained loss')
    plt.title(r'Training Loss over Epochs for pretraining $\beta$-VAE on Unidirectionality dataset')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(False)
    plt.legend()
    plt.savefig('losses-betavae-uni.png')
    plt.show()
