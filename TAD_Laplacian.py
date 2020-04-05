
import numpy as np
from itertools import groupby
import bisect 
from matplotlib import pyplot as plt
from matplotlib import gridspec
import matplotlib.patches as mpatches

class TAD_Laplacian():
    def __init__(self):
        self.max_pixel = None
        #For the first iteration result
        self.fiedler_val_it1 = None
        self.fiedler_vec_it1 = None
        self.coords_it1 = None
        #For recursive result
        self.fiedler_vec = None
        self.coords = None
        self.matrix = None
    
    #Toeplitz factorization matrix
    @staticmethod
    def toeplitz(A, toeplitz_metric):

        E = np.zeros(A.shape)

        for i in range(A.shape[0] - 1):
            old_diag = np.diag(A, k=i)
            old_shape = old_diag.shape[0]
            old_diag = old_diag[old_diag > 0]
            if old_diag.shape[0] != 0:
                if toeplitz_metric == 'num':
                    new_diag = (1 / old_diag.shape[0]) * np.sum(old_diag) * np.ones(old_shape)
                elif toeplitz_metric == 'card':
                    new_diag = (1 / np.unique(old_diag).shape[0]) * np.sum(old_diag) * np.ones(old_shape)
                E += np.diag(new_diag, k=i) + np.diag(new_diag, k=-i)
            else:
                pass

        E = np.where(E == 0, 1, E)
        return E
    
    @staticmethod
    def adjacency_matrix(A, precision, connected=True):
        if precision == 0:
            mask = A
        else:
            mask = np.where(A >= precision, 1, 0)

        N = A.shape[0]
        for i in range(N):
            for j in range(N):
                if i == j:
                    mask[i,j] = 0
                elif (i == j+1 or i == j-1) and connected:
                    mask[i,j] = 1
                else:
                    pass
        return mask
    
    @staticmethod
    def degree_vector(A): #matrix A should be preprocessed
        d = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            d[i] = np.sum(A[i,:])
        return d
    
    @staticmethod
    def fiedler(d, A, norm=False):
        assert type(A) == np.ndarray, 'Matrix A should be type of numpy.ndarray'
        assert A.shape[0] == A.shape[1], 'Matrix A should be square.'
        assert all(d), 'All degrees should be bigger than zero.'

        if norm:
            L = np.diag(d) - A
            L = (np.diag(d**(-1/2))@L@np.diag(d**(-1/2)))
        else:
            L = np.diag(d) - A 

        val, vec = np.linalg.eig(L)
        idx = val.argsort()  
        val = val[idx]
        vec = vec[:,idx]*(-1)
        return np.real(val[1]), vec[:,1]

    #Find coordinates of TADs
    @staticmethod
    def box_coordinates(vec, leftborder):
        arr = np.where(vec<0, 0, 1)
        i = leftborder
        indices = [i]
        for (key, it) in groupby(arr):
            i += len(list(it))
            indices.append(i)
        return indices
    
    #Fit algorithm to a Hi-C matrix
    def fit(self, matrix, precision=0, 
            connected=True, norm_laplacian=True, toeplitz=True, toeplitz_metric='num',         
            recursive=True, minlimit = 10, threshold = 0.01, verbose=False):
        """
        toeplitz_metric :: string, 
        - 'num' - nomalization by count of elements
        - 'card'- normalization by count of unique elements
        """
        #some attributes
        self.max_pixel = matrix.max()
        self.matrix = matrix
        
        if toeplitz:
            matrix = matrix / TAD_Laplacian.toeplitz(matrix, toeplitz_metric=toeplitz_metric)
        A = TAD_Laplacian.adjacency_matrix(matrix, precision=precision, connected=connected)
        d = TAD_Laplacian.degree_vector(A)
        val, vec = TAD_Laplacian.fiedler(d, A, norm=norm_laplacian)
        box_coords = TAD_Laplacian.box_coordinates(vec,0)
        
        self.fiedler_val_it1 = val
        self.fiedler_vec_it1 = vec
        self.coords_it1 = box_coords
        self.fiedler_vec = vec
        self.coords = box_coords
        
        ###Recursive splitting###
        if recursive:
            self.coords = box_coords
            prev_coords = []
            self.fiedler_vec = vec
            j=0
            while True:
                j += 1
                prev_coords = self.coords
                if verbose:
                    print('Iteration:', j) 
                new_coords = []
                for i in range(len(self.coords) - 1):
                    borders = [self.coords[i], self.coords[i+1]]

                    #Check if the submatrix too small
                    if borders[1] - borders[0] <= minlimit:
                        sub_coords = borders
                        new_coords += sub_coords
                        continue

                    #Find Fiedler val and vec for submatrix
                    sub_A = A[borders[0]:borders[1], borders[0]:borders[1]]
                    sub_d = d[borders[0]:borders[1]]
                    subval, subvec = TAD_Laplacian.fiedler(sub_d, sub_A, norm=norm_laplacian)

                    #Check if the submatrix is connected enough
                    if subval >= threshold:
                        sub_coords = borders
                        new_coords += sub_coords
                        continue

                    #Update fiedler vector
                    self.fiedler_vec[borders[0]:borders[1]] = subvec
                    #Find new box coordinates
                    leftborder = borders[0]
                    sub_coords = TAD_Laplacian.box_coordinates(subvec, leftborder)
                    new_coords += sub_coords
                # Update coordinates
                self.coords = list(sorted(set(new_coords)))
                #print('coords:', self.coords)

                if self.coords == prev_coords:
                    #print('Stop')
                    print('Converged in {} iterations'.format(j))
                    break
    
    def visualize(self, poscolor='green', negcolor='blue', 
                  vmax=1, vmin=None, cmap='inferno', transform_func=None):
        ### Start Plotting ###
        plt.figure(figsize=(11,13))
        gs = gridspec.GridSpec(2, 2, height_ratios=[1,8], 
                               width_ratios=[20,1], hspace=0, wspace=0.05)
        ax = plt.subplot(gs[1,0])
        axu = plt.subplot(gs[0,0])
        cbar = plt.subplot(gs[1,1])

        #colors for fiedler vector bars
        axucolors = np.where(self.fiedler_vec<0, negcolor, poscolor)

        #Add boxes on the plot
        box_coords = self.coords
        for i in range(len(box_coords)-1):
            # Create a Rectangle patch
            borders = [box_coords[i], box_coords[i+1]]
            boxwidth = borders[1] - borders[0]
            boxcolor = axucolors[(borders[1] + borders[0])//2]
            rect = mpatches.Rectangle((borders[0],borders[0]),boxwidth, boxwidth, 
                                      linewidth=2, edgecolor=boxcolor, facecolor='none')
            ax.add_patch(rect)    

        #Hi-C map
        if transform_func is not None:
            viz_matrix = transform_func(self.matrix)
            assert viz_matrix.shape == self.matrix.shape, 'Transformed matrix and original one must have equal shapes'
        else:
            viz_matrix = self.matrix
        ax.imshow(viz_matrix, vmax=vmax, vmin=vmin, cmap=cmap)
        ax.xaxis.tick_top()
        ax.set_ylabel('Hi-C map', fontsize=12)
        # colorbar
        pc = plt.pcolor(viz_matrix, cmap=cmap, vmax=vmax)
        plt.colorbar(pc, cax=cbar)
        #Fiedler vector barplot
        x = np.arange(self.matrix.shape[0])
        axu.bar(x, height=self.fiedler_vec, color = axucolors, align='edge')
        axu.set_ylabel('Fiedler\nvector', fontsize = 12)
        axu.set_xlim([0, self.matrix.shape[0]])
        axu.set_xticks([])

        plt.show()
