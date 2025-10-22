import os

DRIVE_PATH = '/global/scratch/users/xinzhou/GraphModel_new/data/'
DRIVE_PYTHON_PATH = DRIVE_PATH.replace('\\', '')


## the space in `My Drive` causes some issues,
## make a symlink to avoid this
SYM_PATH = DRIVE_PATH

from sklearn import preprocessing
#plt.style.use('seaborn-whitegrid')
import numpy as np
import scipy

import torch
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np
from sklearn.linear_model import LogisticRegression
str_y = "no"
str_g = "Contm1_abide"
str_model ="linear"
str_num ="0"
p0 = 0.5
lambda0 = 0.25
n = 0
d =0
 

np.random.seed(123+int(str_num))



num_update_p = 10
batch_size = 64
learning_rate = 1
num_update_theta = 50
r0 = 10
#0
#0


g = np.load(DRIVE_PATH + 'data/data_real_final/graph_'+ str_g + '.npy')
X_label = np.load(DRIVE_PATH + 'data/data_real_final/class_abide.npy')
n = g.shape[0]
d = g.shape[1]


id_shuffle = np.arange(n)
np.random.shuffle(id_shuffle)
X0_all = torch.tensor(g[id_shuffle]).float()
dat_XY = torch.tensor(X_label[id_shuffle,np.newaxis])
X_label = X_label[id_shuffle]
#n = int(n * 0.6)
X0 = X0_all[:n].float()
X = X0[:]
X_all=X0_all[:]






num_iterations = num_update_p * num_update_theta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  GridSearchCV


def compute_matrix_term_and_top_r0_reconstruction(X, p0, r_0):
    """
    Computes the matrix term and reconstructs the top r_0 rank approximation using SVD.

    Parameters:
    X (torch.Tensor): Input matrix of shape [n, d].
    p0 (float or torch.Tensor): Scalar or vector to define P.
    r_0 (int): Number of top singular values, left singular vectors, and right singular vectors to extract.

    Returns:
    torch.Tensor: The reconstructed top-rank approximation of the matrix term.
    """
    # Transpose X back to [d, n] for consistency
    X = X.T  # Now X has shape [d, n]

    # Dimensions
    d, n = X.shape

    # Construct P from p0
    if isinstance(p0, torch.Tensor):
        if p0.numel() != d:  # Check if p0 matches the feature dimension d
            raise ValueError("p0 must have the same number of elements as the feature dimension d")
        P = torch.diag(p0)
    else:
        P = torch.eye(d, device=X.device) * p0  # Scalar p0 applied to diagonal matrix

    # Compute M (d x d)
    term1 = (1 / n) * torch.matmul(X, X.T)
    one_n = torch.ones((n, 1), device=X.device)  # Column vector of ones
    term2 = (1 / n**2) * torch.matmul(X, one_n) @ torch.matmul(one_n.T, X.T)
    M = term1 - term2

    # Compute the off-diagonal elements of M
    off_diag_M = M - torch.diag(torch.diag(M))  # Remove diagonal elements

    # Compute P^2 * off-diagonal(M)
    P_squared = torch.matmul(P, P)
    matrix_term = torch.matmul(P_squared, torch.diag(torch.diag(M))) + off_diag_M

    # Perform SVD on the matrix_term
    U, S, V = torch.svd(matrix_term)  # Full SVD

    # Extract the top r_0 singular values and vectors
    U_r = U[:, :r_0]
    S_r = S[:r_0]
    

    return U_r* torch.sqrt(S_r)


def Aug_loss_lin(A,B, G, tau):
    #print(G.shape)
    n=A.shape[0]
    #positive
    sum1 = torch.sum(torch.sum(A * B, dim=1))
    #negative
    sum2 = torch.sum(A @ B.T) 
    #penalty
    pen = (torch.norm(G.T @ G, 'fro'))**2
    #print("loss0:", -sum1/n + sum2/(n**2))
    #print("penalty0:", pen * tau/8)
    #print(torch.max(G))
    return (-sum1/n + sum2/(n**2)+ pen * tau/8)
    

def Aug_loss(gAX, gAIX, G,tau):
    batch_size = gAX.shape[0]
    cos_sim = torch.nn.CosineSimilarity()(torch.repeat_interleave(gAX, batch_size, dim=0), gAIX.repeat(batch_size,1)).reshape(batch_size,batch_size)/tau
    val_softmax = torch.diag(torch.nn.functional.softmax(cos_sim, dim=1))
    loss = -torch.mean(torch.log(val_softmax))
    return loss



def sample_batch_id(data, batch_size):
  return(random.sample(range(len(data)),batch_size))



def generate_A_upd(p):
    # Initialize a tensor to store the output values
    output = torch.empty_like(p)
    
    # Generate random values to decide the outcome for each entry in `p`
    random_vals = torch.rand(len(p))
    
    # Define conditions based on probabilities
    condition1 = random_vals < (1 - p) / 2
    condition2 = (random_vals >= (1 - p) / 2) & (random_vals < (1 + p) / 2)

    
    # Assign values based on the conditions
    output[condition1] = 0
    output[condition2] = 0.5
    output[~(condition1 | condition2)] = 1
    
    return output
for _method in ["Proposed"]:    
    for str_model in ["linear"]:
        for r in [r0]:
            for tau in [1]:
        
                model = torch.nn.Linear(d, r, bias=False)
                p0_n = torch.tensor([0.0] * d)
    
            loss_linear = 1
            if loss_linear == 1:
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
            elif loss_linear == 0:
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            p0_n = torch.tensor([0.0] * d)

            
    
            # Training loop
            for iteration in range(num_iterations):
                # get batch data

                if loss_linear == 1:
                    batch_X = X * 1.0
                else:
                    batch_id = sample_batch_id(X, batch_size)
                    batch_X = X[batch_id] * 1.0
    
                # Forward pass
                A_diag = generate_A_upd(p0_n)
                AX = batch_X * A_diag.unsqueeze(0)
                AIX = batch_X - AX
                gAX = model(AX)
                gAIX = model(AIX)
                gX = model(batch_X)
                G = model.weight.clone()
               
    
    
                # Calculate the loss
                if loss_linear == 1:
                #if iteration >=  (num_iterations // 2 -1):
                    loss = Aug_loss_lin(gAX, gAIX, G, tau)   
                else:
                    loss = Aug_loss(gAX, gAIX, G, tau)   

    
                # Backpropagation and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if str_model == "linear":
                      iter_N = num_update_theta
    
                if np.abs(iteration -  (num_iterations // 2 -1)) <-5:
                    print([iteration,loss.item()])
    
    
                if (iteration+1) % iter_N == 0:
                    print([iteration,loss.item()])
                    with torch.no_grad():
                        weights = model.weight.clone()  # Shape: (500, 1000)
                        
                        # Step 1: Calculate L2 norm for each column
                        l2_norms = torch.norm(weights, p=2, dim=0)
                        
                        # Step 2: Get the indices of the top 100 columns based on L2 norm
                        _, top100_indices = torch.topk(l2_norms, 150)
                        
                        # Step 3: Create a mask to zero out all but the top 100 columns
                        mask = torch.zeros_like(weights)
                        mask[:, top100_indices] = 1  # Keep only the top 100 columns
                        
                        # Step 4: Apply the mask to the weights (setting other columns to 0)
                        weights *= mask
                        
                        # Step 5: Update the model's weights with the modified tensor
                        model.weight.copy_(weights)
                        if (iteration+1) % num_update_theta == 0:
                            Q = list(model.parameters())[0].clone().T
                            p0_n = torch.sqrt((torch.norm(Q, p=2, dim=1)**2) / torch.var(X, dim=0, unbiased=False))
                            p0_n = torch.nan_to_num(p0_n, nan=1, posinf=1, neginf=1)
                            p0_n = torch.clip(p0_n,min=0,max=1)

                            

                            np.save(DRIVE_PATH + "/res_Sim/Q_all_"+ str_g + "_"+ str(str_y) + str_num +"_"+ str(r) +"_"+ str(tau) +"_"+ str(batch_size) +"_"+ str(learning_rate) +"_"+ str(iteration) +"_"+ str(p0) +"_"+ str(lambda0) +"_" + str_model+ ".npy",
                                   Q)
                            





