# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

# Define the QVAE model class
class QVAE(nn.Module):
    def __init__(self, n_qubits, q_depth):
        super(QVAE, self).__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth

        # Initialize quantum circuit parameters
        self.q_params = nn.Parameter(torch.randn(q_depth, n_qubits, 3))

        # Define the quantum device and cuda 
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Classical layers to preprocess the image before sending to quantum circuits
        self.pre_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # [batch_size, 16, 32, 32]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # [batch_size, 32, 16, 16]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 8, 8]
            nn.ReLU(),
            nn.Flatten(),  # Flatten to [batch_size, 64 * 8 * 8]
            nn.Linear(64 * 8 * 8, n_qubits),  # Reduce dimensionality to match n_qubits
            nn.ReLU()
        )

        # Classical post-processing layers for the encoder
        self.fc1 = nn.Linear(n_qubits, 2*latent_dim)

        # Classical post-processing layers for the decoder
        self.fc2 = nn.Linear(n_qubits, 64 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.Unflatten(dim=1, unflattened_size=(64, 8, 8)),  # Reshape to [batch_size, 64, 8, 8]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # [batch_size, 32, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # [batch_size, 16, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),  # [batch_size, 3, 64, 64]
            nn.Sigmoid()  # Output values in [0, 1]
        )

    def quantum_encoder(self, x):
        batch_size = x.size(0)

        @qml.qnode(self.dev, interface="torch")
        def circuit(x, params):
            qml.templates.AngleEmbedding(x, wires=range(self.n_qubits))
            qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        quantum_enc = []
        for i in range(batch_size):
            quantum_enc.append(circuit(x[i], self.q_params))
        
        return torch.Tensor(quantum_enc)

    def quantum_decoder(self, z):
        batch_size = z.size(0)

        @qml.qnode(self.dev, interface="torch")
        def circuit(z, params):
            qml.templates.AngleEmbedding(z, wires=range(self.n_qubits))
            qml.templates.StronglyEntanglingLayers(params, wires=range(self.n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        quantum_dec = []
        for i in range(batch_size):
            quantum_dec.append(circuit(z[i], self.q_params))

        return torch.Tensor(quantum_dec)

    def encode(self, x):
        # Preprocess image with classical layers
        x = self.pre_encoder(x)
        quantum_enc = self.quantum_encoder(x)
        h1 = torch.relu(self.fc1(quantum_enc.float().to(self.device)))
        mu, logvar = torch.chunk(h1, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        quantum_dec = self.quantum_decoder(z)
        h4 = torch.relu(self.fc2(quantum_dec.float().to(self.device)))
        return self.decoder(h4)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, z, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar):
        # Reconstruction loss
        recon_loss = nn.MSELoss()(recon_x, x)

        # KL divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss
        return recon_loss + kld_loss

# Define Diffusion model class
class DiffusionModel(nn.Module):
    def __init__(self, input_channels, timesteps, latent_dim, device="cuda"):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.latent_dim = latent_dim
        self.input_channels = input_channels
        self.device = device

        # Beta scheduling (for diffusion)
        self.betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
        self.alpha = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        # Encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, latent_dim)  # Output: [batch, latent_dim*2] for mean and log_var
        )

        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(256, 8, 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward_diffusion(self, x, t):
        """ Forward process: add noise to the data and then encode. """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t]).view(-1, 1, 1, 1)
        noise = torch.randn_like(x)
        x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        
        # Encode the noisy image to latent space
        z = self.encoder(x)
        return z

    def reverse_diffusion(self, z, t):
        """ Reverse process: reconstruct the data from noisy latent. """
        x_reconstructed = self.decoder(z)
        for i in reversed(range(t)):
            beta_t = self.betas[i]
            noise = torch.randn_like(x_reconstructed)
            x_reconstructed = x_reconstructed - torch.sqrt(beta_t) * noise
        return x_reconstructed

    def encode(self, x):
        """ Encode the input into a lower-dimensional latent space (2D). """
        z = self.forward_diffusion(x, t=self.timesteps - 1)
        return z

    def decode(self, z):
        """ Decode the 2D latent representation back to the original 3D/4D space. """
        x_reconstructed = self.reverse_diffusion(z, t=self.timesteps - 1)
        return x_reconstructed

    def forward(self, x):
        """ Forward pass for encoding and decoding. """
        z = self.encode(x)
        x_reconstructed = self.decode(z)
        return x_reconstructed, z

# Define IWAE model class
class IWAE(nn.Module):
    def __init__(self, n_qubits, nc, beta, num_samples):
        super(IWAE, self).__init__()
        self.n_qubits = n_qubits
        self.beta = beta
        self.num_samples = num_samples

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, n_qubits)
        self.fc_logvar = nn.Linear(1024, n_qubits)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),  # 4x4 -> 512 channels
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output values in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, num_samples):
        batch_size = x.size(0)
        
        x = x.unsqueeze(1).expand(-1, num_samples, -1, -1, -1).reshape(batch_size * num_samples, *x.size()[1:])
        
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded).view(batch_size, num_samples, -1)
        logvar = self.fc_logvar(encoded).view(batch_size, num_samples, -1)
        
        z = self.reparameterize(mu, logvar).view(batch_size * num_samples, -1)

        z = torch.tanh(z)
        
        # Ensure the decoder output matches [batch_size * num_samples, 3, 64, 64]
        reconstructed = self.decoder(z)
        
        return reconstructed.view(batch_size, num_samples, *x.size()[1:]), mu, logvar, z

    @staticmethod
    def iwae_loss_function(reconstructed, x, mu, logvar, beta, num_samples):
        batch_size = x.size(0)
        
        # Reshape the input tensor to match the expected batch_size and num_samples
        x = x.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
        
        # Compute Reconstruction Loss (MSE)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='none')
        reconstruction_loss = reconstruction_loss.view(batch_size, num_samples, -1).sum(dim=-1)
        
        # Compute KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        # IWAE Loss (log-sum-exp trick for numerical stability)
        log_weight = -reconstruction_loss - beta * kl_divergence
        max_log_weight = log_weight.max(dim=1, keepdim=True)[0]
        weight = torch.exp(log_weight - max_log_weight)

        # Compute Importance Weighted Loss
        loss = -max_log_weight.squeeze() - torch.log(weight.mean(dim=1) + 1e-10)
        return loss.mean()

# Define disentangled beta VAEs model class
class DisentangledBetaVAE(nn.Module):
    def __init__(self, latent_dim, beta):
        super(DisentangledBetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 32, 32]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # [batch_size, 128, 16, 16]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # [batch_size, 256, 8, 8]
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # [batch_size, 512, 4, 4]
            nn.ReLU(),
            nn.Flatten(),  # Flatten to [batch_size, 512 * 4 * 4]
            nn.Linear(512 * 4 * 4, 2*latent_dim),  # [batch_size, 256]
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, n_qubits)
        self.fc_logvar = nn.Linear(1024, n_qubits)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512 * 4 * 4),  # [batch_size, 512 * 4 * 4]
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),  # [batch_size, 512, 4, 4]
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # [batch_size, 256, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # [batch_size, 128, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # [batch_size, 64, 32, 32]
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # [batch_size, 3, 64, 64]
            nn.Sigmoid()  # Output values in [0, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')

        # KL divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Total loss with beta
        return recon_loss + self.beta * kld_loss

# Define the VAE class
class BetaVAE(nn.Module):
    def __init__(self, n_qubits, nc, beta):
        super(BetaVAE, self).__init__()
        self.n_qubits = n_qubits
        self.beta = beta
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU()
        )

        self.fc_mu = nn.Linear(1024, n_qubits)
        self.fc_logvar = nn.Linear(1024, n_qubits)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_qubits, 512 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(dim=1, unflattened_size=(512, 4, 4)),  # 4x4 -> 512 channels
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Sigmoid()  # Output values in [0, 1]
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        z = torch.tanh(self.reparameterize(mu, logvar))
        reconstructed = self.decoder(z)
        return reconstructed, mu, logvar, z

    @staticmethod
    def loss_function(reconstructed, x, mu, logvar, beta):
        # Reconstruction loss (MSE or BCE)
        reconstruction_loss = F.mse_loss(reconstructed, x, reduction='sum')
        # KL Divergence
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Combine losses with beta factor
        loss = reconstruction_loss + beta * kl_divergence
        return loss

# Define the Classical Discriminator
class Discriminator(nn.Module):
    def __init__(self, n_qubits, label_dims):
        super(Discriminator, self).__init__()
        self.l1 = nn.Linear(label_dims, n_qubits, bias=False)
        self.model = nn.Sequential(
            nn.Linear(2*n_qubits, 128, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64, bias=False),
        )
        # GAN head for real/fake classification
        self.gan_output = nn.Linear(64, 1)

        # Physics head for predicting the 4-dimensional physics parameters
        self.physics_output = nn.Linear(64, 4)

    def forward(self, inputs, label):
        x1 = inputs
        x2 = self.l1(label)
        combined_input = torch.cat((x1, x2), dim=1)
        x = F.leaky_relu(self.model(combined_input), 0.2)
        gan_pred = torch.sigmoid(self.gan_output(x))
        physics_pred = self.physics_output(x)
        return gan_pred.view(-1), physics_pred

# Define Quantum Generator class
class QuantumGenerator(nn.Module):
    def __init__(self, n_qubits, q_depth, n_generators):
        super(QuantumGenerator, self).__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.n_generators = n_generators
        
        # Initialize the quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Initialize the weights for the orthogonal transformation
        self.q_params = nn.ParameterList([
            nn.Parameter(nn.init.orthogonal_(torch.rand(n_qubits, n_qubits) * 0.02 - 0.01),   requires_grad=True)
            for _ in range(n_generators)
        ])

        
        # Initialize the biases
        self.bias = nn.ParameterList([
            nn.Parameter(torch.Tensor(n_qubits).uniform_(-0.01, 0.01), requires_grad=True)
            for _ in range(n_generators)
        ])
        
        # Linear layer to map to the final output size
        #self.fc = nn.Linear(n_generators * n_qubits, n_qubits)

   
    def efficient_su2_entanglement(self, params):
        """Parameterized Efficient SU2 with Circular Entanglement."""
        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i % self.n_qubits)

        # Circular entanglement
        for i in range(self.n_qubits):
            qml.CNOT(wires=[i, (i + 1) % self.n_qubits])

        for i in range(self.n_qubits):
            qml.RY(params[i], wires=i % self.n_qubits)

    def quantum_generator(self, params):
        """Define the QNode."""
        @qml.qnode(self.dev, interface='torch')
        def circuit():
            # Efficient SU2-like ansatz with circular entanglement
            for _ in range(self.q_depth):
                self.efficient_su2_entanglement(params)

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]      
        
        return circuit()

    def forward(self, x):
        batch_sizes = x.size(0) # Get the batch size
        batch_results = []

        for i in range(batch_sizes): # Loop over the batch size
            element_results = []
            for W, b in zip(self.q_params, self.bias):
                # Apply orthogonal transformation
                theta = torch.matmul(x[i], W) + b
                output = self.quantum_generator(theta) # Pass each batch element through the quantum generator
                element_results.append(torch.tensor(output, dtype=torch.float32, device=x.device))
            batch_results.append(torch.cat(element_results, dim=0))  # Concatenate results for this element

        x = torch.stack(batch_results) # Stack to reform the batch dimension
        
        return x
