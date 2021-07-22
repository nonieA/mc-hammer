from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from math import cos,sin, acos, pi
from random import uniform, seed

def quadratic(v,d,c):
    a = v
    b = 2 * v * c
    c = v * c**2 -d**2
    x1 = (-b+(b**2 - 4*a*c)**0.5)/(2*a)
    x2 = (-b-(b**2 - 4*a*c)**0.5)/(2*a)
    return (x1,x2)

def normal_find(d,v):
    x = (d**2/v)**0.5
    return x

def make_many_blobs(k,dim,n,sd,centers):
    blobs_list = []
    labs_list = []
    for i in range(k):
        blobs = make_blobs(
            n_samples=n[i],
            n_features=dim,
            centers=[tuple(centers[0])],
            cluster_std=sd[i]
        )[0]
        blobs = blobs + centers[i]
        blobs_list.append(blobs)
        labs = np.array([i for j in range(n[i])])
        labs_list.append(labs)
    blobs_full = np.concatenate(blobs_list)
    labs_full = np.concatenate(labs_list)
    return blobs_full,labs_full

def get_centers(theta,hyp,mult,dim,cos1st = True):
    if cos1st:
        d_coords = [hyp*cos(theta),hyp*sin(theta)] *mult
    else:
        d_coords = [hyp * sin(theta),hyp * cos(theta)] * mult
    coords = np.array(d_coords[:dim])
    return(coords)

def generate_data(k,dimx,sep,n,sd,se,noise):
    dim = round(dimx*(1-noise))
    noise_dim = dimx - dim
    if isinstance(sep,list) ==False:
        sep = [sep for i in range(k)]
    if isinstance(n,list) ==False:
        n = [n for i in range(k)]
    if isinstance(sd,list) ==False:
        sd = [sd for i in range(k)]
    seed(se)
    if dim % 2 == 0:
        mult = int(dim / 2)
    else:
        mult = int(round(dim / 2)) + 1
    if k == 2:
        side_ab = sd[0]*sep[0] + sd[1]*sep[1]
        theta = uniform(0,0.25*pi)

        ce_b = [side_ab*sin(theta),side_ab*cos(theta)] * mult
        ce_b = np.array(ce_b[:dim])
        centers = [np.array([0]*dim),ce_b]
    if k == 3:
        side_ab = sd[0]*sep[0] + sd[1]*sep[1]
        side_ac = sd[0]*sep[0] + sd[2]*sep[2]
        side_bc = sd[1]*sep[1] + sd[2]*sep[2]
        theta_r = uniform(0.1,1/4*pi)
        theta_a = (side_ab**2 + side_ac**2 - side_bc**2)/(2*side_ac*side_ab)
        theta_x = theta_r+theta_a
        ce_b = get_centers(theta_x,side_ab,mult,dim)
        ce_c = get_centers(theta_r,side_ac,mult,dim)
        centers = [np.array([0]*dim),ce_b,ce_c]

    if k == 4:
        side_ab = sd[0]*sep[0] + sd[1]*sep[1]
        side_ac = sd[0]*sep[0] + sd[2]*sep[2]
        side_bc = sd[1]*sep[1] + sd[2]*sep[2]
        side_ad = sd[0]*sep[0] + sd[3]*sep[3]
        side_bd = sd[1]*sep[1] + sd[3]*sep[3]
        theta_a = (side_ab**2 + side_ac**2 - side_bc**2)/(2*side_ac*side_ab)
        theta_ad = (side_ab**2 + side_ad**2 - side_bd**2)/(2*side_ad*side_ab)
        theta_r = uniform(0.1,1/4*pi)
        theta_x = theta_a +  theta_r
        theta_y = 1/2 * pi - (theta_x +theta_ad)
        ce_b = get_centers(theta_x, side_ab, mult, dim)
        ce_c = get_centers(theta_r, side_ac, mult, dim)
        ce_d = get_centers(theta_y, side_ac, mult, dim,False)
        centers = [np.array([0]*dim),ce_b,ce_c,ce_d]
    blobs_full, labs_full = make_many_blobs(k, dim, n, sd, centers)
    centers = np.array(centers)/blobs_full.max()
    blobs_full = blobs_full/blobs_full.max()
    noise_array = np.random.rand(sum(n),noise_dim)
    noise_cent_list = []
    for i in np.unique(labs_full):
        one_clust = noise_array[labs_full == i, :]
        noise_cent_list.append(one_clust.mean(axis=0))
    noise_cent_list = np.array(noise_cent_list)
    centers = np.concatenate((centers,noise_cent_list),axis = 1)
    blobs_full = np.append(blobs_full,noise_array, axis =1)
    return blobs_full,labs_full,centers

def plot_dims(dim1,dim2,blobs_full,labs_full):
    for i in range(max(labs_full) + 1):
        one_b = blobs_full[labs_full == i,:]
        plt.scatter(one_b[:,dim1],one_b[:,dim2])
    plt.show()




if __name__ == '__main__':
    blobs_full,labs_full = generate_data(3,4,4,100,2,2)
    blobs_full, labs_full = generate_data(4, 2, 2, 100, 2, 2)
    plot_dims(0,1,blobs_full,labs_full)
    dim1 = 0
    dim2 = 1
    for i in range(max(labs_full) + 1):
        one_b = blobs_full[labs_full == i,:]
        plt.scatter(one_b[:,dim1],one_b[:,dim2])
    plt.scatter(centers[:,dim1],centers[:,dim2])