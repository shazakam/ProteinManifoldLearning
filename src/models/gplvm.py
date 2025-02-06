import numpy as np 
import torch
import torch.nn as nn
import gpytorch
class GPLVM:

    def __init__(self, Y : torch.tensor, latent_dim, kernel_params):
        self.Y = Y
        self.YY_T = torch.mm(Y, torch.transpose(Y, 0, 1)) # N x N
        self.latent_dim = latent_dim # Dimensionality of latent space
        self.kernel_params = kernel_params # Kernel Parameters must be tensors with requires_grad=True
        self.N = Y.shape[0] # Number of data points
        self.D = Y.shape[1] # Dimensionality of data points
        self.X = self.initialise_X() # Initial Latent space points
        
        
    def fit(self, num_iter, lr):
        optimiser = torch.optim.SGD([self.X] + self.kernel_params, lr=lr)
        for iteration in range(num_iter):

            # Compute NLL Loss and use PyTorch SGD to update X and kernel parameters
            optimiser.zero_grad()
            neg_log_likelihood = self.compute_neg_log_likelihood()
            neg_log_likelihood.backward()
            optimiser.step()
            print(f"Iteration {iteration + 1}/{num_iter}, NLL Loss: {neg_log_likelihood.item()}")
            

    def compute_neg_log_likelihood(self):
        K = self.rbf_kernel(self.X, self.X, self.kernel_params)
        
        K_inv = torch.inverse(K + torch.eye(self.N)*1e-6)
        term1 = torch.abs(torch.logdet(K))
        # term2 = torch.trace(torch.mm(K_inv, self.YY_T))
        term2 = torch.trace(torch.mm(torch.mm(self.Y.T, K_inv), self.Y))
        print(term1, term2)
        neg_log_likelihood = self.D*term1/2 + term2/2
   
        return neg_log_likelihood

    def rbf_kernel(self, X1, X2, kernel_params):
        """
        Compute the RBF kernel between two sets of inputs.

        Args:
            X1 (torch.Tensor): First input tensor of shape (N, D).
            X2 (torch.Tensor): Second input tensor of shape (M, D).
            length_scale (float): Length scale parameter of the RBF kernel.
            variance (float): Variance parameter of the RBF kernel.

        Returns:
            torch.Tensor: Kernel matrix of shape (N, M).
        """
        length_scale = kernel_params[0]
        # Compute the squared Euclidean distance between each pair of points
        sqdist = torch.cdist(X1, X2, p=2) ** 2
        print(sqdist)
        # Compute the RBF kernel
        K = torch.exp(-0.5 * (sqdist / length_scale) ** 2)
        
        return K
    
    def initialise_X(self):
        return torch.randn(self.N, self.latent_dim, requires_grad=True)

        
