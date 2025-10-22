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
str_g = "simContSparse1LinUpd"
str_model ="linear"
str_num ="0"
p0 = 0.5
lambda0 = 0.25
n = 250
d =990
 

np.random.seed(123+int(str_num))



num_update_p = 10
batch_size = 64
learning_rate = 1
num_update_theta = 500
r0 = 10
mag_QZ = 1.25
noise_sd = 0

n_patient = n
str_g0 = str_g[:]
str_g = str_g0 + "_" + str(mag_QZ)+ "_" + str(noise_sd)
num_iterations = num_update_p * num_update_theta

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
    return (-sum1/n + sum2/(n**2)+ pen * tau/8)/(d)
    

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
    
for noise_sd in [0,2,4,6]:
    print("############################################")
    print("noise sd is", noise_sd)
    n = n_patient 
    str_g = str_g0 + "_" + str(mag_QZ)+ "_" + str(noise_sd)
    
    
    def generate_diagonal_covariance_matrix(d, const = 0.1):
        # Create the diagonal elements (1/d, 2/d, ..., d/d)
        diagonal_elements = np.arange(1, d + 1) / d
        # Create a diagonal matrix using these elements
        covariance_matrix = np.diag(diagonal_elements ** 2)
        return covariance_matrix
    
    import numpy as np
    
    # Define the dimensions and parameters
    r = r0   # Example dimension, replace with actual value
    
    
    binary_vector = np.zeros(d)
    binary_vector[np.random.choice(d, 50, replace=False)] = 1
    c_0 = binary_vector
    supp_true = np.where(c_0==1)[0]
    # Generate the Q matrix
    Q_true_ori = np.random.normal(size=(d, r))
    Q_true= np.diag(c_0) @ Q_true_ori  * (mag_QZ)



    
    
    # Generate the Z matrix
    Z_true = np.random.normal(size=(r, n)) 
    
    
    
    # Create the label matrix X_label
    X_label = ((Z_true[0]-Z_true[1])>0).astype(int)
    
    # Compute QZ
    QZ = (Q_true @ Z_true) 
    
    QZ = QZ.T
    
    cov_matrix = generate_diagonal_covariance_matrix(d,const=0.05)
    
    # Generate the noise matrix xi
    xi = np.random.multivariate_normal(np.zeros(d), cov_matrix*(noise_sd**2), n)
    # Compute g
    g = QZ + xi

    
    
    
    
    str_g = str_g + "_"+ str(n) + "_"+ str(d) + "_"+ str(r0) + "_1"
    n = int(n * 0.6)
    X0 = torch.tensor(g[:n]).float()
    X0_all = torch.tensor(g[:]).float()
    dat_XY = torch.tensor(X_label[:,np.newaxis])
    X = X0[:]
    X_all=X0_all[:]
    
    
    
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    import warnings
    warnings.filterwarnings("ignore")
    
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import  GridSearchCV

    
    
    
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

                            
                            Q_oracle = Q_true 
                            print("dist:", torch.norm(Q @ Q.T - Q_oracle @ Q_oracle.T, p='fro').item())
                            top50_indices = torch.topk(l2_norms, 50)
                            print("Edge Acc:",iteration+1,len(np.intersect1d(supp_true, top50_indices))/len(supp_true))
                            np.save(DRIVE_PATH + "/res_Sim/Iden_"+ str_g + "_"+ str(str_y) + str_num +"_"+ str(r) +"_"+ str(tau) +"_"+ str(batch_size) +"_"+ str(learning_rate) +"_"+ str(iteration) +"_"+ str(p0) +"_"+ str(lambda0) +"_" + str_model+ ".npy",
                                   len(np.intersect1d(supp_true, top50_indices))/len(supp_true))
                            np.save(DRIVE_PATH + "/res_Sim/Q_"+ str_g + "_"+ str(str_y) + str_num +"_"+ str(r) +"_"+ str(tau) +"_"+ str(batch_size) +"_"+ str(learning_rate) +"_"+ str(iteration) +"_"+ str(p0) +"_"+ str(lambda0) +"_" + str_model+ ".npy",
                                   Q)
                            
                            Z = np.dot(np.linalg.inv(np.dot(Q.T, Q)), np.dot(Q.T, X_all.detach().numpy().T)).T
                            Z_all = preprocessing.scale(Z)
                            
                            Z_train = Z_all[:n]
                            Z_test = Z_all[n:]
                        
                            y_train = dat_XY[:n,0].numpy()
                            y_test = dat_XY[n:,0].numpy()
                        
                        
                            # Use the best parameters to train the final classifier
                            svm_classifier = SVC(kernel='linear', C=1, random_state = 1)
                        
                        
                            # Train the classifier
                            svm_classifier.fit(Z_train, y_train)
                        
                            # Make predictions
                            y_pred = svm_classifier.predict(Z_test)
                        
                            # Evaluate accuracy
                            accuracy = accuracy_score(y_test, y_pred)
                            print("Accuracy:", accuracy)
                        
                            np.save(DRIVE_PATH + "/res_Sim/acc_"+ str_g + "_"+ str(str_y) + str_num +"_"+ str(r) +"_"+ str(tau) +"_"+ str(batch_size) +"_"+ str(learning_rate) +"_"+ str(iteration) +"_"+ str(p0) +"_"+ str(lambda0) +"_" + str_model+ ".npy",accuracy )
                        


    import numpy as np
    
    def sparse_rank_1(X, u1, v1, s1, lambda_, iter_max, tol):
        """
        Sparse rank-one matrix approximation.
    
        Parameters:
            X: np.ndarray of shape (n, p) - input matrix
            u1: np.ndarray - initial rank-1 approximation (typically from SVD)
            v1: np.ndarray - initial rank-1 approximation (typically from SVD)
            s1: float - initial scalar value from SVD
            lambda_: float - parameter controlling sparsity
            iter_max: int - maximum number of iterations
            tol: float - tolerance for stopping criteria
    
        Returns:
            u: np.ndarray - sparse rank-1 approximation of X
            v: np.ndarray - sparse rank-1 approximation of X
            s: float - scalar from the approximation
        """
        u = u1
        v_old = v1 * s1
        v = np.zeros_like(v1)
    
        lambda_scaled = lambda_ / np.abs(v_old)
        p = len(v1)
    
        for j in range(iter_max):
            y = X.T @ u
            for n in range(p):
                v[n] = np.sign(y[n]) * (np.abs(y[n]) >= lambda_scaled[n]) * (np.abs(y[n]) - lambda_scaled[n])
            
            u = X @ v / (np.linalg.norm(X @ v, 2) + 1e-7)
            delta = np.linalg.norm(v - v_old, 2)
            
            if delta < tol:
                break
                
            v_old = v
    
        v = v / np.linalg.norm(v, 2)
        s = u.T @ X @ v
    
        return u, v, s
    
    def sparse_pca(X, lambda_, K, iter1, iter2, tol1, tol2):
        """
        Perform PCA matrix decomposition with consistent sparsity pattern across all principal components.
    
        Parameters:
            X: np.ndarray of shape (n, p) - data matrix
            lambda_: float - regularization parameter
            K: int - number of principal components to estimate
            iter1: int - max iterations for standard sparse PCA
            iter2: int - max iterations for globally sparse PCA
            tol1: float - tolerance for standard sparse PCA
            tol2: float - tolerance for globally sparse PCA
    
        Returns:
            U1: np.ndarray - principal components from standard sparse PCA
            V1: np.ndarray - loadings from standard sparse PCA
            U2: np.ndarray - principal components from globally sparse PCA
            V2: np.ndarray - loadings from globally sparse PCA
        """
        n, p = X.shape
        U, S, V = np.linalg.svd(X, full_matrices=False)
        
    
    
        # Initialize for globally sparse PCA
        U2 = U[:, :K]
        V2 = (V.T)[:, :K]
    

    
        lambda1 = lambda_ / np.linalg.norm(V2, axis=1)
        
    
        U2_old = U2.copy()
        for j in range(iter2):
            for i in range(p):
                res_i = U2.T @ X[:, i]
                if (lambda1[i] * np.sqrt(p)) < 2 * np.linalg.norm(res_i):
                    v = (1 - (lambda1[i] * np.sqrt(p) / (2 * np.linalg.norm(res_i)))) * res_i
                else:
                    v = np.zeros(K)
                V2[i, :] = v
            
            U_j, _, V_j = np.linalg.svd(X @ V2)
            U2 = U_j[:, :K] @ V_j.T
    
            delta = np.linalg.norm(U2 - U2_old, 'fro')
            if delta < tol2:
                break
            
            U2_old = U2.copy()
    
        return U2, V2
    
    def predict_sparse_pca(X, V, K):
    
        U_j, _, V_j = np.linalg.svd(X @ V)
        U = U_j[:, :K] @ V_j.T
    
        return U    
    # Parameters for Sparse PCA
    
    iter1 = 100
    iter2 = 100
    tol1 = 1e-4
    tol2 = 1e-4
    for lambda_ in [0.001]:
        # Perform Sparse PCA
        U2, V2 = sparse_pca(X.detach().numpy()[:,:], lambda_, r, iter1, iter2, tol1, tol2)
        l1_norms = np.sum(np.abs(V2), axis=1)
        supp_true = np.where(c_0==1)[0]
        supp_gspca = np.argsort(l1_norms)[-50:][::-1]
        print("PCA Iden:", len(np.intersect1d(supp_true, supp_gspca))/len(supp_true))
        np.save(DRIVE_PATH + "/res_Sim/Iden_"+ str_g + "_"+ str(str_y) + str_num +"_"+ str(r) +"_"+ str(tau) +"_"+ str(batch_size) +"_"+ str(learning_rate) +"_"+ str(iteration) +"_"+ str(p0) +"_"+ str(lambda0) +"_" + "PCA"+ ".npy",
                len(np.intersect1d(supp_true, supp_gspca))/len(supp_true))

        Z_all = predict_sparse_pca(X_all, V2, r)
        Z_all = preprocessing.scale(Z_all)
        Z_train = Z_all[:n]
        Z_test = Z_all[n:]
        y_train = dat_XY[:n,0].numpy()
        y_test = dat_XY[n:,0].numpy()
        
        
        
        
        
        # Use the best parameters to train the final classifier
        svm_classifier = SVC(kernel='linear', C=1, random_state = 1)
        svm_classifier.fit(Z_train, y_train)
        
        
        # Train the classifier
        svm_classifier.fit(Z_train, y_train)
        
        # Make predictions
        y_pred = svm_classifier.predict(Z_test)
        
        # Evaluate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print("PCA Acc:", accuracy)
        
        np.save(DRIVE_PATH + "/res_Sim/acc_"+ str_g + "_"+ str(str_y) + str_num +"_"+ str(r) +"_"+ str(tau) +"_"+ str(batch_size) +"_"+ str(learning_rate) +"_"+ str(iteration) +"_"+ str(p0) +"_"+ str(lambda0) +"_" + "PCA" + ".npy",accuracy )

    print("############################################")
    
