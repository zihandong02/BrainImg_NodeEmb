import os
from sklearn.decomposition import SparsePCA

DRIVE_PATH = 
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
from sklearn.decomposition import PCA


import time

# Record start time
start_time = time.time()
max_dif=0
for int_num0 in range(50):
    #int_num0=26
    str_num = str(int_num0)
    str_model ="linear"
    str_y = "no"
    str_g = "SimContSparse"
    p0 = 0.5
    lambda0 = 0.25
    n = 250
    d = 990
     
    
    np.random.seed(123+int(str_num))
    
    
    
    
    batch_size = 64
    learning_rate = 0.01
    num_iterations = 50001
    r0 = 10
    mag_QZ = 1.25
    noise_sd = 4
    str_g = str_g + "_" + str(mag_QZ)+ "_" + str(noise_sd) 
    
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

    #g = np.random.binomial(n=1, p=Mu)

    
    
    str_g = str_g + "_"+ str(n) + "_"+ str(d) + "_"+ str(r0) + "_1"
    n = int(n * 0.6)
    X0 = torch.tensor(g[:n]).float()
    X0_all = torch.tensor(g[:]).float()
    dat_XY = torch.tensor(X_label[:,np.newaxis])
    X = X0[:]
    X_all=X0_all[:]
    tau=0.5

    pca = PCA()
    #pca.fit(X.detach().numpy())
    pca.fit(X)
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_ * 100
    cur_dif = explained_variance_ratio[9]-explained_variance_ratio[10]
    if max_dif < cur_dif:
        max_dif = cur_dif
        ind_best = int_num0
        X_best = X[:]
print(ind_best)
end_time = time.time()

# Calculate time elapsed
elapsed_time = end_time - start_time
#print(f"Elapsed time: {elapsed_time} seconds")


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Assuming X is your data matrix
pca = PCA()
#pca.fit(X.detach().numpy())
pca.fit(X_best)
# Get the explained variance ratio
explained_variance_ratio = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_/explained_variance_ratio.sum()*100
cumulative_explained_variance = explained_variance_ratio.cumsum()

# Create an index for the components (PC1, PC2, etc.)
components = range(1, len(explained_variance_ratio) + 1)



# Plotting
sns.set(style="whitegrid")
plt.figure(figsize=(8, 6))
sns.lineplot(x=components, y=explained_variance_ratio, marker='o')
plt.axvline(x=r, color='red', linestyle='--')
plt.gca().get_xaxis().set_visible(False)


# Labels and title
plt.xlabel('Principal Component')
plt.ylabel('Percentage of Explained Variance')
#plt.title('Cumulative Explained Variance by Principal Component')
plt.xticks(components)  # Ensure x-ticks match component numbers
plt.grid(True)

# Show plot
#plt.show()
plt.savefig(f'Cont_{noise_sd}.png') 