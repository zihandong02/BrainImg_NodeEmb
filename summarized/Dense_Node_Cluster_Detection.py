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
str_g = "simContNodeLinUpd"
str_model ="linear"
str_num ="0"
p0 = 0.5
lambda0 = 0.25
n = 250
d =210
 

np.random.seed(123+int(str_num))



num_update_p = 20
batch_size = 64
learning_rate = 0.1
num_update_theta = 500
r0 = 10
mag_QZ = 5
noise_sd = 0
n_patient = n
str_g0 = str_g[:]
str_g = str_g0 + "_" + str(mag_QZ)+ "_" + str(noise_sd)
num_iterations = num_update_p * num_update_theta

def get_tilde_Q(k, r):
    output = np.random.normal(size=(k, k, r))
    for i in range(k-1):
        for j in range(i+1, k):
            output[i, j, :] = output[j, i, :]
    return output

def get_indices(d0):
    indices = []
    for i in range(1, d0+1):
        for j in range(1, d0+1):
            if i > j:
                indices.append((i, j))
    return indices

def gen_AR_cor_mat(k, rho):
    if not -1 < rho < 1:
        raise ValueError("rho must be in the range (-1, 1) for positive definiteness.")
    
    # Create the correlation matrix
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            R[i, j] = rho ** abs(i - j)
    
    return R
def gen_AR_cov_mat(k, rho, eps = 0.1):
    correlation_matrix = gen_AR_cor_mat(k, rho)
    norms = np.sqrt(np.random.random(size=(k))+ eps) 
    D = np.diag(norms)
    covariance_matrix = D @ correlation_matrix @ D  # Element-wise scaling by norms
    
    return covariance_matrix

def gen_Q(d0, k, r):
    
    
    # Distribute nodes into clusters, ensuring all nodes are assigned
    node_set = np.zeros(d0, dtype=int)
    n = d0 // k
    remainder = d0 % k
    
    start = 0
    for i in range(1, k + 1):
        end = start + n + (1 if i <= remainder else 0)
        node_set[start:end] = i
        start = end
    
    np.random.shuffle(node_set)
    
    indices = get_indices(d0)
    d = len(indices)
    Q = np.random.normal(size=(d, r))
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    C = gen_AR_cov_mat(k,0.5)
    #C_Low = 0.5
    #C = C_Low + (1-C_Low) * np.random.random(size=(k))
    #C = 2 * np.random.random(size=(k)) - 1
    #C2 = 2 * np.random.random(size=(k)) - 1
    #C3 = 2 * np.random.random(size=(k)) - 1

    
    for i in range(d):
        Q[i, :] = Q[i, :] * C[node_set[indices[i][0] - 1] - 1,node_set[indices[i][1] - 1] - 1]
        #Q[i, :] = Q[i, :] * (
        #    C[node_set[indices[i][0] - 1] - 1] * C[node_set[indices[i][1] - 1] - 1] +
        #    C2[node_set[indices[i][0] - 1] - 1] * C2[node_set[indices[i][1] - 1] - 1] +
        #    C3[node_set[indices[i][0] - 1] - 1] * C3[node_set[indices[i][1] - 1] - 1]
        #)
        #if node_set[indices[i][0] - 1] == node_set[indices[i][1] - 1]:
        #    Q[i, :] = Q[i, :] * C[node_set[indices[i][0] - 1] - 1] 
        #else:
        #    Q[i, :] = Q[i, :] * C[node_set[indices[i][0] - 1] - 1] * C[node_set[indices[i][1] - 1] - 1]
        
        
    
    return {'Q': Q, 'node_cluster': node_set, 'edge': indices, 'clus_weight': C}


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


from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import rand_score
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse.linalg import eigsh
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

def get_clustering(S, k, n_clusters):
    """
    Perform clustering using the top k eigenvectors of a matrix S.

    Args:
        S (numpy.ndarray): The input square matrix (e.g., similarity matrix).
        k (int): The number of top eigenvectors to use.
        n_clusters (int): Number of clusters for k-means.

    Returns:
        labels (numpy.ndarray): Cluster labels for each index.
    """
    # Step 1: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    
    # Step 2: Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    print(eigenvalues)
    top_k_eigenvectors = eigenvectors[:, sorted_indices[:k]]
    
    # Step 3: Normalize the rows of the eigenvector matrix
    normed_vectors = top_k_eigenvectors 
    
    # Step 4: Perform clustering on the normalized eigenvector rows
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(normed_vectors)
    
    return labels

def spectral_clustering(S, k):
    # Step 1: Compute the degree matrix
    D = np.diag(S.sum(axis=1))
    
    # Step 2: Compute the unnormalized Laplacian matrix
    L = D - S
    
    # Step 3: Compute the k smallest eigenvalues and eigenvectors
    eigvals, eigvecs = eigsh(L, k=k, which='SM')
    
    # Step 4: Use the eigenvectors to form the U matrix
    U = eigvecs
    
    # Step 5: Run k-means on the rows of U
    kmeans = KMeans(n_clusters=k, random_state=123)
    labels = kmeans.fit_predict(U)
    
    return labels

def get_label4(Q, n_clus = 2, clus_weight=None,clus=None):
    num_nodes = int(np.sqrt(Q.shape[0]*2+0.25)+0.5)
    
    u = np.linalg.norm(Q, axis=1)
    
    S = np.zeros((num_nodes, num_nodes))
    i0=-1
    for j1 in range(1,num_nodes):
      for j2 in range(j1):
          i0 += 1
          S[j1, j2] = u[i0]
          S[j2, j1] = u[i0]

    if clus_weight is not None:
        np.fill_diagonal(S, clus_weight[clus-1] ** 2)

    print(S)
    
    labels = get_clustering(S, n_clus, n_clus)

    return(labels)

def get_label0(Q, n_clus = 2):
    num_nodes = int(np.sqrt(Q.shape[0]*2+0.25)+0.5)
    
    u = np.linalg.norm(Q, axis=1)
    
    S = np.zeros((num_nodes, num_nodes))
    i0=-1
    for j1 in range(1,num_nodes):
      for j2 in range(j1):
          i0 += 1
          S[j1, j2] = u[i0]
          S[j2, j1] = u[i0]


    print(S)
    
    labels = get_clustering(S, n_clus, n_clus)

    return(labels)
    
def get_label(Q, n_clus = 2):
    num_nodes = int(np.sqrt(Q.shape[0]*2+0.25)+0.5)
    
    u = np.linalg.norm(Q, axis=1)
    
    S = np.zeros((num_nodes, num_nodes))
    i0=-1
    for j1 in range(1,num_nodes):
      for j2 in range(j1):
          i0 += 1
          S[j1, j2] = u[i0]
          S[j2, j1] = u[i0]
    
    #labels = spectral_clustering(S, n_clus)
    do_clustering = SpectralClustering(n_clusters=len(np.unique(cluster_true)),
                                     assign_labels='kmeans',
                                     random_state=0).fit(S)
    labels = do_clustering.labels_
    return(labels)

def get_label2(Q, n_clus = 2):
  num_nodes = int(np.sqrt(Q.shape[0]*2+0.25)+0.5)
  U, _, _ = np.linalg.svd(Q, full_matrices=False)
  u = U[:, 0]

  S = np.zeros((num_nodes, num_nodes))
  i0=-1
  for j1 in range(1,num_nodes):
      for j2 in range(j1):
          i0 += 1
          S[j1, j2] = u[i0]
          S[j2, j1] = u[i0]

  P, _, _ = np.linalg.svd(S)
  kmeans = KMeans(n_clusters=n_clus).fit(P[:, :1])
  return(kmeans.labels_)

    
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
    
    
    d0 = int(np.sqrt(2*d + 0.25) + 0.5)
    
    Q_res = gen_Q(d0, 3, r)
    Q_true_ori = Q_res['Q']
    clus = Q_res['node_cluster']
    indices = Q_res['edge']
    clus_weight = Q_res['clus_weight']
    Q_true= Q_true_ori  * (mag_QZ) 



    
    
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
    
    
                if (iteration+1) % iter_N == 0:
                    print([iteration,loss.item()])
                    with torch.no_grad(): 
                        if (iteration+1) % num_update_theta == 0:
                            Q = list(model.parameters())[0].clone().T
                            p0_n = torch.sqrt((torch.norm(Q, p=2, dim=1)**2) / torch.var(X, dim=0, unbiased=False))
                            p0_n = torch.nan_to_num(p0_n, nan=1, posinf=1, neginf=1)
                            p0_n = torch.clip(p0_n,min=0,max=1)

                            
                            cluster_true = clus.copy()
    
                            cluster_est = get_label(Q, n_clus = len(np.unique(cluster_true)))
                            rand_Q = rand_score(cluster_est,cluster_true)
                            print("rand: ", rand_Q)
                        
                            np.save(DRIVE_PATH + "/res_Sim/rand_"+ str_g + "_"+ str(str_y) + str_num +"_"+ str(r) +"_"+ str(tau) +"_"+ str(batch_size) +"_"+ str(learning_rate) +"_"+ str(iteration) +"_"+ str(p0) +"_"+ str(lambda0) +"_" + str_model+ ".npy",rand_Q  )
                        

    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import  GridSearchCV
    # Step 1: Instantiate PCA with the desired number of components (r)
    r = r0  # Example value for the reduced dimensionality
    pca = PCA(n_components=r, random_state = 1)
    
    # Step 2: Fit PCA on the training data
    pca.fit(X.T)

    cluster_est = get_label(pca.transform(X.T), n_clus = len(np.unique(cluster_true)))
    rand_Q = rand_score(cluster_est,cluster_true)
    print("PCA rand: ", rand_Q)
    np.save(DRIVE_PATH + "/res_Sim/rand_"+ str_g + "_"+ str(str_y) + str_num +"_"+ str(r) +"_"+ str(tau) +"_"+ str(batch_size) +"_"+ str(learning_rate) +"_"+ str(iteration) +"_"+ str(p0) +"_"+ str(lambda0) +"_" + "PCA" + ".npy",rand_Q)


    print("############################################")