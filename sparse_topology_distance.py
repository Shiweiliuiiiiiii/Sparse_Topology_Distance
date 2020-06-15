"""
Official implementation of Topological Insights in Sparse Neural Networks with Graph Theory of Liu et al.
The compared sparse topologies should be saved as .nzp format by "scipy.sparse.save_npz".
# If you use parts of this code please cite the following articles:
#@article{Mocanu2018SET,
#  author =        {Mocanu, Decebal Constantin and Mocanu, Elena and Stone, Peter and Nguyen, Phuong H. and Gibescu, Madeleine and Liotta, Antonio},
#  journal =       {Nature Communications},
#  title =         {Scalable Training of Artificial Neural Networks with Adaptive Sparse Connectivity inspired by Network Science},
#  year =          {2018},
#  doi =           {10.1038/s41467-018-04316-3}
#}
"""
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import pickle as pickle
import time
import numpy as np
import scipy.sparse as ss
from multiprocessing import Process, Manager
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys
import os
import argparse

################################################################
# Parameters
################################################################
parser = argparse.ArgumentParser(description='Spars topology distance')
parser.add_argument('--epoch', default=0, type=int, help='Topology graph of which epoch to measure')
parser.add_argument('--layer', default=1, type=int, help='Topology graph of which layer to measure')
parser.add_argument('--hidden_size', default=784, type=int, help='Hidden size of the network.')
parser.add_argument('--nprocessor', default=7, type=int, help='Number of processors to process calculation')
parser.add_argument('--topo_root', metavar='DIR', default='topo/', help='path to topology (default: topo/)')
parser.add_argument('--plot_path', metavar='DIR', default='plot/', help='path to save topology distance heatmap (default: plot)')

args = parser.parse_args()

if not os.path.exists(args.plot_path):
    os.mkdir(args.plot_path)

LAYER = args.layer
################################################################
# Main
################################################################
def main():
    # File to save calculated topology distance matrix
    DATAFILE = 'Distance_e5_set_l%.d_epoch%.3d' % (args.layer, args.epoch)
    compute(args.epoch, DATAFILE)
    imshow(args.epoch, DATAFILE)
################################################################
# Sparse toplogy distance functions
################################################################
def compute(epoch, DATAFILE):
    all_epochs_dirs = sorted([f for f in listdir(args.topo_root) if not isfile(join(args.topo_root, f))])

    print("Network name:",all_epochs_dirs)
    matrices = sorted(
        [args.topo_root + e + '/' + 'epoch_%.3d/' % epoch + f for e in all_epochs_dirs for f in
         listdir(args.topo_root + e + '/' + 'epoch_%.3d/' % epoch) if
         (isfile(args.topo_root + e + '/'+ 'epoch_%.3d/' % epoch + f) and '.npz' in f and 'l' + str(LAYER) in f)])
    print('Matrices:', matrices)

    mana = Manager()
    data = mana.dict({})
    # exit(1)

    saved_data = mana.dict(load_obj(DATAFILE))

    def proc_run(counter, saved_data, datad, indices):
        c1 = 0
        print('Process ' + str(counter) + ' started with ' + str(len(indices)) + ' computations! Let\'s go!')
        t1 = time.time()
        for (i1, i2) in indices:
            c = NNSTD(matrices[i1], matrices[i2], saved_data)
            datad[(i1, i2)] = np.mean(c)
            datad[(i2, i1)] = np.mean(c)
            c1 += 1
            print(counter,
                  "Progress:", {100 * int(1000 * c1 / (len(indices))) / 1000},
                   format_time((time.time() - t1) / c1, len(indices) - c1), "left")
            sys.stdout.flush()

    # multi processor opti
    num_procs = args.nprocessor
    print("Starting to compute with",num_procs,"processors...")
    indices = [[] for _ in range(num_procs)]
    c2 = 0
    for i1, m1 in enumerate(matrices):
        for i2, m2 in enumerate(matrices):
            if i2 <= i1:
                indices[c2 % num_procs].append((i1, i2))
                c2 += 1
    ps = []
    for i in range(num_procs):
        p = Process(target=proc_run, args=(i, saved_data, data, indices[i]))
        ps.append(p)
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    sys.stdout.flush()
    save_obj(dict(saved_data), DATAFILE)

def getNormalizedEditDistance(a, b):

    # for two hidden neuron trees, return 1 if they are completely different, 0 if they are exactly the same
    if (len(a) == 0 and len(b) == 0):
        return 0
    intersect, inda, indb = np.intersect1d(a, b, return_indices=True)
    if (intersect.shape[0] > 0):
        edit = np.delete(a, inda).shape[0] + np.delete(b, indb).shape[0]
        normalizedEdit = edit / np.unique(np.concatenate((a, b), axis=0)).shape[0]
    else:
        normalizedEdit = 1
    return normalizedEdit


def compareLayers(layer1, layer2, saved_data={}, name1="", name2=""):
    try:
        # Assume symmetric
        col_ind, minCost = saved_data[(name2, name1)]
        row_ind = list(range(len(col_ind)))
    except KeyError:
        try:
            col_ind, minCost = saved_data[(name1, name2)]
            row_ind = list(range(len(col_ind)))
        except KeyError:
            bottomList1 = []
            for j in range(layer1.shape[1]):
                bottomList1.append(np.nonzero(layer1[:, j])[0])
            bottomList2 = []
            for j in range(layer2.shape[1]):
                bottomList2.append(np.nonzero(layer2[:, j])[0])

            editMatrix = np.zeros((layer1.shape[1], layer2.shape[1]))
            for j1 in range(layer1.shape[1]):
                for j2 in range(layer2.shape[1]):
                    editMatrix[j1, j2] = getNormalizedEditDistance(bottomList1[j1], bottomList2[j2])

            row_ind, col_ind = linear_sum_assignment(editMatrix)
            minCost = editMatrix[row_ind, col_ind].sum()

            if name1 != '': saved_data[(name1, name2)] = col_ind, minCost
    return minCost / layer1.shape[1], row_ind, col_ind


def compareNetworkSemiTopological(weightsNN1, weightsNN2, saved_data={}, names1=[], names2=[]):
    numLayers = len(weightsNN1)
    assert len(weightsNN2) == numLayers
    editDistance = np.zeros(numLayers)
    k = 0
    layerNN1 = weightsNN1[k].copy()
    layerNN2 = weightsNN2[k].copy()
    try:
        n1, n2 = names1[k], names2[k]
    except:
        n1, n2 = '', ''

    editDistance[k], hid_layerNN1, hid_layerNN2 = compareLayers(layerNN1, layerNN2, saved_data, n1, n2)
    for k in range(1, numLayers):
        layerNN1 = weightsNN1[k].copy()
        layerNN2 = weightsNN2[k].copy()
        try:
            n1, n2 = names1[k], names2[k]
        except:
            n1, n2 = '', ''
        for j in range(weightsNN2[k].shape[0]):
            layerNN2[j, :] = weightsNN2[k][hid_layerNN2[j], :].copy()
        editDistance[k], hid_layerNN1, hid_layerNN2 = compareLayers(layerNN1, layerNN2, saved_data, n1, n2)
    return editDistance

def NNSTD(topo1, topo2, saved_data={}):
    '''
    topo1: sparse topology matrice saved by .npz format.
    topo2: sparse topology matrice saved by .npz format.
    Note: Two topologies compared should have the same shape after converting to dense matrices.
    '''
    def get_mats(topo):
        def get_mat(layer, layermax):
            new_topo = topo.replace('_l' + str(layermax) + '.', '_l' + str(layer) + '.')
            mat = ss.load_npz(new_topo).A[:-1, :]
            mat[mat > 0] = 1
            mat[mat < 0] = 1
            return mat, new_topo

        layer_max = int(topo[-5])
        layer = 0
        mat = []
        names = []

        while (layer <= layer_max):
            m, n = get_mat(layer, layer_max)
            mat.append(m)
            names.append(n)
            layer = layer + 1
        return mat, names

    mat1, names1 = get_mats(topo1)
    mat2, names2 = get_mats(topo2)

    s = compareNetworkSemiTopological(listtonp(mat1), listtonp(mat2), saved_data, names1, names2)
    return s

def format_time(seconds_one, amount_left):
    exa = 1.05
    return str(int(exa * seconds_one * amount_left / (60 * 60))) + "h " + str(
        int(exa * seconds_one * amount_left / 60) % (60)) + "min " + str(
        int(exa * seconds_one * amount_left % 60)) + "s"

def save_obj(obj, name):
    with open('./save/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    try:
        with open('./save/' + name + '.pkl', 'rb') as f:
            dic = pickle.load(f)
            print("Loading {name} with {len(dic.keys())} keys... Last ten: {list(dic.keys())[-10:]}")
            return dic
    except FileNotFoundError:
        print("WARNING: did not find {name}. Cancel compute (CTRL-C/CTRL-Z) to avoid overwriting.")
        time.sleep(3)
        return {}

def listtonp(listin):
    a = []
    for item in listin:
        a.append(np.array(item))
    return a

def rename(matrice_name):
    # used for the xticks and yticks for heatmaps
    name = matrice_name[11:13]
    return name
################################################################
# Topology Visualization
################################################################
def imshow(epoch, DATAFILE):
    all_epochs_dirs = sorted([f for f in listdir(args.topo_root) if not isfile(args.topo_root + f)] )
    print("Network name:",all_epochs_dirs)
    matrices = sorted(
        [args.topo_root + e + '/' + 'epoch_%.3d/' % epoch  + f for e in all_epochs_dirs for f in listdir(args.topo_root + e + '/' + 'epoch_%.3d/' % epoch) if
         (isfile(args.topo_root + e + '/' + 'epoch_%.3d/' % epoch  + f) and '.npz' in f and 'l' + str(LAYER) in f)])

    print('Length', len(matrices))
    print("matrices:", matrices)
    print("Heatmap with {len(matrices) ** 2} pixels...")
    print(matrices[:6], '... len', len(matrices))

    mana = Manager()

    saved_data = mana.dict(load_obj(DATAFILE))

    data = []
    hidden_size = args.hidden_size
    for m1 in matrices:
        data.append([])
        for m2 in matrices:
            pixel = []
            for interm_layer in range(LAYER + 1):
                m1_n = m1.replace('l' + str(LAYER) + '_', 'l' + str(interm_layer) + '_')
                m2_n = m2.replace('l' + str(LAYER) + '_', 'l' + str(interm_layer) + '_')
                try:
                    v = saved_data[(m1_n, m2_n)][1]
                    pixel.append(v)
                except KeyError:
                    try:
                        v = saved_data[(m2_n, m1_n)][1]
                        pixel.append(v)
                    except:
                        raise AssertionError
            print(m1, m2, pixel)
            data[-1].append(np.mean(pixel) / (hidden_size))
    data = np.array(data)

    plt.figure(figsize=(4, 4.4))
    ax = plt.gca()
    im = ax.imshow(data, interpolation='none', cmap=plt.get_cmap('gnuplot'),vmax=1,vmin=0,)#gnuplot
    plt.xticks(list(range(len(matrices))), labels=[rename(s) for s in matrices], rotation='vertical')
    plt.yticks(list(range(len(matrices))), labels=[rename(s) for s in matrices])
    plt.title('Layer ' + str(LAYER))
    plt.xlabel('Density')
    plt.ylabel('Density')
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.tight_layout()

    plt.savefig(args.plot_path +  DATAFILE + '.pdf')
    plt.show()

if __name__ == '__main__':
    main()
