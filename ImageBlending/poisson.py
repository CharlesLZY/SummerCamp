import numpy as np
from scipy import sparse
from scipy.sparse import linalg

def map_Omega(src,dst,mask):
    h,w,_ = src.shape

    coordinate_map = []
    index_map = np.zeros([h,w])
    idx = 0
    for i in range(h):
        for j in range(w):
            if mask[i,j,0] == 255:
                coordinate_map.append((i,j))
                index_map[i,j] = idx
                idx += 1
    return index_map, coordinate_map


### construct Discrete Poisson Equation
def construct_DPE(src, dst, mask, index_map, coordinate_map, mixing=False):
    h,w,c = src.shape
    N = len(coordinate_map)
    A = np.zeros([N, N], dtype=np.float16)
    b = np.zeros([N, 3])
    
    for i in range(N):
        A[i,i] = 4
        y,x = coordinate_map[i]

        if not mixing: 
            # 4. * is very important, force it to be float
            b[i, :] = 4. * src[y, x, :] - src[y - 1, x, :] - src[y + 1, x, :] - src[y,x - 1, :] - src[y, x + 1, :] # calculate divergence according to guiding gradient v
            
        else: ### mixing gradient
            if abs(np.mean(src[y, x, :]) - np.mean(src[y - 1, x, :])) > abs(np.mean(dst[y, x, :]) - np.mean(dst[y - 1, x, :])):
                b[i, :] += 1. * src[y, x, :] - src[y - 1, x, :] # 1. * is very important, force it to be float
            else:
                b[i, :] += 1. * dst[y, x, :] - dst[y - 1, x, :] # 1. * is very important, force it to be float
            if abs(np.mean(src[y, x, :]) - np.mean(src[y + 1, x, :])) > abs(np.mean(dst[y, x, :]) - np.mean(dst[y + 1, x, :])):
                b[i, :] += 1. * src[y, x, :] - src[y + 1, x, :] # 1. * is very important, force it to be float
            else:
                b[i, :] += 1. * dst[y, x, :] - dst[y + 1, x, :] # 1. * is very important, force it to be float
            if abs(np.mean(src[y, x, :]) - np.mean(src[y, x - 1, :])) > abs(np.mean(dst[y, x, :]) - np.mean(dst[y, x - 1, :])):
                b[i, :] += 1. * src[y, x, :] - src[y, x - 1, :] # 1. * is very important, force it to be float
            else:
                b[i, :] += 1. * dst[y, x, :] - dst[y, x - 1, :] # 1. * is very important, force it to be float
            if abs(np.mean(src[y, x, :]) - np.mean(src[y, x + 1, :])) > abs(np.mean(dst[y, x, :]) - np.mean(dst[y, x + 1, :])):
                b[i, :] += 1. * src[y, x, :] - src[y, x + 1, :] # 1. * is very important, force it to be float
            else:
                b[i, :] += 1. * dst[y, x, :] - dst[y, x + 1, :] # 1. * is very important, force it to be float
                                
        ### check neighbours
        # up
        if y > 0 and mask[y - 1, x, 0] == 255: # Omega
            A[i, int(index_map[y-1,x])] = -1
        else: # partial Omega (boundary)
            b[i,:] += dst[y-1,x,:]
        # down
        if y < h - 1 and mask[y + 1, x, 0] == 255: # Omega
            A[i, int(index_map[y+1,x])] = -1
        else: # partial Omega (boundary)
            b[i,:] += dst[y+1,x,:]
        # left
        if x > 0 and mask[y, x - 1, 0] == 255: # Omega
            A[i, int(index_map[y,x-1])] = -1
        else: # partial Omega (boundary)
            b[i,:] += dst[y,x-1,:]
        # right
        if x < w - 1 and mask[y, x + 1, 0] == 255: # Omega
            A[i, int(index_map[y,x+1])] = -1
        else: # partial Omega (boundary)
            b[i,:] += dst[y,x+1,:]

    return A,b


### solve Discrete Poisson Equation in rgb channels respectively
def solve_DPE(A,b):
    ### Too slow !!!
    # xr,_,_,_ = np.linalg.lstsq(A,b[:,0], rcond=None) # https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.linalg.lstsq.html
    # xg,_,_,_ = np.linalg.lstsq(A,b[:,1], rcond=None)
    # xb,_,_,_ = np.linalg.lstsq(A,b[:,2], rcond=None)

    # A is a sparse matrix with a few 4 and -1 and tons of 0. Therefore, for better computing performance, use the scipy.sparse 
    A = sparse.lil_matrix(A, dtype=int) # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
    # Conjugate Gradient Iteration
    xr,_ = linalg.cg(A, b[:, 0]) #  
    xg,_ = linalg.cg(A, b[:, 1])
    xb,_ = linalg.cg(A, b[:, 2])
    return xr,xg,xb















