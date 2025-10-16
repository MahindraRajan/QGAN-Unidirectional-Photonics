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
from models import IWAE, Discriminator, QuantumGenerator

# Adaptive noise
def relu(x):
    return x * (x > 0)
def get_noise_upper_bound(gen_loss, disc_loss, original_ratio):
    R = disc_loss.detach().cpu().numpy()/gen_loss.detach().cpu().numpy()
    return np.pi/2 + (5 *np.pi / 2) * relu(np.tanh((R - (original_ratio))))

# Load the pretrained IWAE and train the QGAN
def train_qgan(generator, discriminator, iwae, num_samples, excelDataTensor, dataloader, optimizerG, optimizerD, criterion, num_epochs):
    generator.train()
    discriminator.train()
    iwae.eval()  # Pretrained IWAE in eval mode
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
            label = torch.full((b_size * num_samples,), real_label, device=device)

            with torch.no_grad():  # Use the pretrained IWAE
                recon_real, mu, logvar, latent_real = iwae(real_cpu, num_samples)

            # Populate noise2 and noise
            for j in range(b_size):
                excelIndex = x * b_size + j  # Ensuring we get the correct index
                try:
                    gotdata = excelDataTensor[excelIndex]
                except IndexError:
                    break

                # Create noise2 and noise by replicating gotdata for num_samples
                for _ in range(num_samples):
                    tensorA = gotdata.view(1, label_dims)  # Adjusted to 4 values
                    noise2 = torch.cat((noise2, tensorA), 0)

                    # Concatenate features with random noise
                    tensor1 = torch.cat((gotdata, torch.rand(latent)), dim=0).view(1, n_qubits)
                    noise = torch.cat((noise, tensor1), 0)

            noise = noise.to(device)
            noise2 = noise2.to(device)

            # Generate fake latent space features with generator
            fake_latent = generator.forward(noise).to(device)

            # Train Discriminator
            output_real = discriminator.forward(latent_real, noise2).view(-1)
            errD_real = criterion(output_real, label)
            errD_real.backward(retain_graph=True)

            label.fill_(fake_label)
            output_fake = discriminator.forward(fake_latent, noise2).view(-1)
            errD_fake = criterion(output_fake, label)
            errD_fake.backward(retain_graph=True)
            errD = errD_real + errD_fake     # Add the gradients from the all-real and all-fake batches
            optimizerD.step()

            # Update Generator
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator.forward(fake_latent, noise2).view(-1)
            errG = criterion(output, label)
            errG.backward(retain_graph=True)
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

    print("QGAN Training Completed with Pretrained IWAE.")


# Testing function to check the generator's output using testTensor
def test_generator(generator, iwae, testTensor, device):
    generator.eval()
    iwae.eval()

    with torch.no_grad():
        fake_latent = generator(testTensor)
        fake_images = iwae.decoder(fake_latent)
        fake = fake_images.detach().cpu()
                
    img_list.append(vutils.make_grid(fake, nrow=10, padding=2, normalize=True))

    return img_list

def Excel_Tensor(spectra_path):
    # Location of excel data
    excelData = pd.read_csv(spectra_path, header = 0, index_col = 0)    
    excelDataSpectra = excelData.iloc[:,:5] #index until the last point of the spectra in the Excel file
    excelDataTensor = torch.tensor(excelDataSpectra.values).type(torch.FloatTensor)
    return excelData, excelDataSpectra, excelDataTensor

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
    num_samples = 5
    num_epochs = 50
    workers = 1
    lrG = 1e-4  # Increase the learning rate for the generator
    lrD = 1e-4  # Keep the learning rate for the discriminator lower
    label_dims = 5
    img_list = []
    G_losses = []
    D_losses = []
    latent = 4
    beta = 2.0
    # Beta1 hyperparam for Adam optimizers
    beta1 = 0.5
    beta2 = 0.999
    # Establish convention for real and fake labels during training
    real_label = random.uniform(0.9,1.0)  # One-sided label smoothing for real labels
    fake_label = 0  # Fake label remains the same
    # Adaptive noise parameters
    noise_upper_bound = np.pi/8
    original_ratio = None
    upper_bounds = [np.pi/8]

    # Define device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data preparation (Adjust the path as needed)
    img_path = 'C:/....../Training Images-___.csv/'
    spectra_path = 'C:/....../Unidirectional-Training Dataset-___.csv'

    excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)

    dataset = dset.ImageFolder(root=img_path,
                               transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor()
                               ]))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

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

    # Load the pretrained IWAE from a specific directory
    iwae = IWAE(n_qubits = n_qubits, nc = nc, beta = beta, num_samples = num_samples).to(device)
    iwae.load_state_dict(torch.load('pretrained_iwae-uni.pth'))

    # Define optimizers
    optimizerG = optim.Adam(generator.parameters(), lr=lrG, betas=(beta1, beta2))
    optimizerD = optim.Adam(discriminator.parameters(), lr=lrD, betas=(beta1, beta2))
    criterion = nn.BCELoss(reduction='mean')

    start_time = time.time()
    local_time = time.ctime(start_time)
    print('Start Time = %s' % local_time)

    # Train the QGAN
    train_qgan(generator, discriminator, iwae, num_samples, excelDataTensor, dataloader, optimizerG, optimizerD, criterion, num_epochs)

    local_time = time.ctime(time.time())
    print('End Time = %s' % local_time)
    run_time = (time.time()-start_time)/3600
    print('Total Time Lapsed = %s Hours' % run_time)
    
    # Save the final models
    torch.save(generator.state_dict(), 'final_generator_qgan_iwae-uni.pth')
    torch.save(discriminator.state_dict(), 'final_discriminator_qgan_iwae-uni.pth')

    # Test the generator using testTensor
    fixed_batch_size = 100
    testTensor = torch.Tensor()
    generator.load_state_dict(torch.load('final_generator_qgan_iwae-uni.pth'))

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
    img_list = test_generator(generator, iwae, testTensor, device)
    
    # Plot and save G and D Training Losses
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training for Unidirectional dataset")
    plt.plot(G_losses,label="Generator Loss")
    plt.plot(D_losses,label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('losses-qgan-iwae-uni.png')
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
    plt.savefig('fake-and-real-iwae-uni.png')
    plt.show()
