import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset
import torchvision.utils as vutils 
import pandas as pd
import random
import time
from models.models import BetaVAE, Discriminator, QuantumGenerator


# Load the pretrained VAE and train the QGAN
def train_qgan(generator, discriminator, betavae, excelDataTensor, dataloader, optimizerG, optimizerD, criterion, num_epochs):
    generator.train()
    discriminator.train()
    betavae.eval()  # Pretrained VAE in eval mode
    x = 0
    noise = torch.Tensor()
    noise2 = torch.Tensor()
    print("Starting Training Loop...")

    for epoch in range(num_epochs):
        x = 0
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)

            with torch.no_grad():  # Use the pretrained VAE
                recon_real, mu, logvar, latent_real = betavae(real_cpu)

            noise2 = torch.empty(0, label_dims).to(device)
            noise = torch.empty(0, n_qubits).to(device)

            # Use the preloaded excelDataTensor in the training loop
            for j in range(b_size):
                excelIndex = x * batch_size + j
                try:
                    gotdata = excelDataTensor[excelIndex]
                except IndexError:
                    break

                tensorA = gotdata.view(1, label_dims).to(device)  # Adjusted to 4 values
                noise2 = torch.cat((noise2, tensorA), 0)

                # Concatenate features with random noise
                tensor1 = torch.cat((gotdata, torch.rand(latent)), dim=0).view(1, n_qubits).to(device)
                noise = torch.cat((noise, tensor1), 0)

            
            # Generate fake latent space features with G
            fake_latent = generator(noise).to(device)

            # Train Discriminator
            output_real = discriminator(latent_real.detach(), noise2.detach()).view(-1)
            errD_real = criterion(output_real, label.detach())
            errD_real.backward(retain_graph=True)

            label.fill_(fake_label)
            output_fake = discriminator(fake_latent.detach(), noise2).view(-1)
            errD_fake = criterion(output_fake, label)
            errD_fake.backward()

            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            optimizerD.step()

            # Update Generator
            generator.zero_grad()
            label.fill_(real_label)

            output = discriminator(fake_latent, noise2).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f'
                      % (epoch+1, num_epochs, i, len(dataloader),
                         errD.item(), errG.item()))
            
            x += 1
            noise = torch.Tensor()
            noise2 = torch.Tensor()
    
    print("QGAN Training Completed with Pretrained VAE.")

# Testing function to check the generator's output using testTensor
def test_generator(generator, betavae, testTensor, device):
    generator.eval()
    betavae.eval()

    with torch.no_grad():
        fake_latent = generator(testTensor)
        fake_images = betavae.decoder(fake_latent)
        fake = fake_images.detach().cpu()
                
    img_list.append(vutils.make_grid(fake, nrow=10, padding=2, normalize=True))

    return img_list

def Excel_Tensor(spectra_path):
    # Location of excel data
    excelData = pd.read_csv(spectra_path, header = 0, index_col = 0)    
    excelDataSpectra = excelData.iloc[:,:5] #index until the last point of the spectra in the Excel file
    excelDataTensor = torch.tensor(excelDataSpectra.values).type(torch.FloatTensor)
    return excelData, excelDataSpectra, excelDataTensor

# Custom weights initialization called on discriminator
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)



# Main script
if __name__ == '__main__':
    # Parameters
    n_generators = 1
    n_qubits = 9
    q_depth = 2
    nc = 3
    image_size = 64
    batch_size = 18
    num_epochs = 50
    workers = 1
    lrG = 1e-4
    lrD = 1e-4
    label_dims = 5
    img_list = []
    G_losses = []
    D_losses = []
    latent = 4
    beta = 2.0
    # Beta1 hyperparameters for Adam optimizers
    beta1 = 0.5
    beta2 = 0.999
    # Establish conventions for real and fake labels during training
    real_label = random.uniform(0.9,1.0)  # One-sided label smoothing for real labels
    fake_label = 0  # Fake label remains the same

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data preparation (Adjust the path as needed)
    img_path = '/dgxb_home/se21pphy004/Multiclass_Metasurface/Uni_data/Images/'
    spectra_path = '/dgxb_home/se21pphy004/Multiclass_Metasurface/Uni_data/Unidirectional-Training Dataset.csv'

    excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)
    
    dataset = dset.ImageFolder(root=img_path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor()
                               ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=workers, drop_last=True)

    # Plot some training images
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:128], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
    
    # Initialize Quantum Generator and Discriminator
    generator = QuantumGenerator(n_qubits, q_depth, n_generators).to(device)
    discriminator = Discriminator(n_qubits, label_dims).to(device)

    # Apply the weights_init function to randomly initialize all weights
    discriminator.apply(weights_init)

    # Load the pretrained VAE from a specific directory
    betavae = BetaVAE(n_qubits = n_qubits, nc = nc, beta = beta).to(device)
    betavae.load_state_dict(torch.load('pretrained_beta_vae_uni.pth'))

    # Define optimizers
    optimizerG = optim.Adam(generator.parameters(), lr=lrG, betas=(beta1, beta2))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lrD, betas=(beta1, beta2))
    criterion = nn.BCELoss()

    start_time = time.time()
    local_time = time.ctime(start_time)
    print('Start Time = %s' % local_time)

    # Train the QGAN
    train_qgan(generator, discriminator, betavae, excelDataTensor, dataloader, optimizerG, optimizerD, criterion, num_epochs)

    local_time = time.ctime(time.time())
    print('End Time = %s' % local_time)
    run_time = (time.time()-start_time)/3600
    print('Total Time Lapsed = %s Hours' % run_time)
    
    # Save the final models
    torch.save(generator.state_dict(), 'final_generator_qgan_betavae_uni.pth')
    torch.save(discriminator.state_dict(), 'final_discriminator_qgan_betavae_uni.pth')

    # Test the generator using testTensor
    fixed_batch_size = 100
    testTensor = torch.Tensor()
    generator.load_state_dict(torch.load('final_generator_qgan_betavae_uni.pth'))

    for i in range(fixed_batch_size):
        index = i * int(np.floor(len(excelDataSpectra) / fixed_batch_size))
        excel_data = excelDataTensor[index]
        random_noise = torch.rand(latent)
        fixed_noise1 = torch.cat((excel_data, random_noise))
        fixed_noise2 = fixed_noise1.unsqueeze(1)
        fixed_noise = fixed_noise2.permute(1, 0)
        testTensor = torch.cat((testTensor, fixed_noise), 0)

    # Move testTensor to the appropriate device (GPU/CPU)
    testTensor = testTensor.to(device)
    
    # Call the test function
    img_list = test_generator(generator, betavae, testTensor, device)
    
    # Plot and save G and D Training Losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses,label="Generator Loss")
    plt.plot(D_losses,label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses-qgan-betavae_uni.png')
    plt.show()

    # Plot the real images
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    # Plot the fake images from the last epoch
    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("Fake Images")
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.savefig('fake-and-real-betvae-uni.png')
    plt.show()