import math
import copy
import symop
import numpy as np


def apply_basis(str_list):
    """
    apply lattice vector to get Cartesian coordinate for lattice points
    :param str_list: list of structure metadata generated by parse_str function
    :return: str_list with 'LatPnt' in Cartesian coordinate
    """
    cart_str_list = copy.deepcopy(str_list)
    for i in range(len(str_list)):
        str_dict = str_list[i]
        for j in range(len(str_dict['LatPnt'])):
            cart_str_list[i]['LatPnt'][j] = np.dot(np.transpose(str_dict['LatPnt'][j]), str_dict['LatVec'])

    return cart_str_list


def calc_dist(pnt1, pnt2):
    """
    calculate the distance between two point in 3D Cartesian coordinate
    :param pnt1: list
    :param pnt2: list
    :return: dist
    """
    dist = np.linalg.norm(np.array(pnt1) - np.array(pnt2))

    return dist


def scale_clust(clust):
    """
    Transform from unscaled Cartesian coordinates to scaled ones for a given cluster
    :param clust: cluster from clust_list in the formate of [[coord_list], [dist_list], [spin]]
    :return: clust
    """
    size = len(clust[0])
    max_dist = 0
    scaled_clust = copy.deepcopy(clust)
    for i in range(size):
        for j in range(i, size):
            dist = calc_dist(clust[0][i], clust[0][j])
            if max_dist < dist:
                max_dist = dist
    if max_dist != 0:
        scale = max(clust[1]) / max_dist
    else:
        scale = 0
    for i in range(size):
        for j in range(3):
            scaled_clust[0][i][j] = scale * clust[0][i][j]

    return scaled_clust


def frac_to_cart(frac_coord, basis):
    """
    Transform from fraction/direct coord to Cartesian coord for a given structure
    :param frac_coord: list in the format of [0,0,0]
    :param basis: list in the format of [[0.0, 0.0, 3.6], [0.0, 3.6, 0.0], [3.6, 0.0, 0.0]]
    :return: cart_coord
    """
    a1 = np.array(basis[0])
    a2 = np.array(basis[1])
    a3 = np.array(basis[2])
    frac_coord = np.array(frac_coord)
    trans_matr = np.vstack([a1, a2, a3]).T
    # inv_matr = np.linalg.inv(trans_matr)
    cart_coord = np.matmul(trans_matr, frac_coord.T).T

    return list(cart_coord)


def cart_to_frac(cart_coord, basis):
    """
    transform from Cartesian coordinate to direct/fraction coordinate
    :param cart_coord: list in the format of [0, 0, 0]
    :param basis: list in the format of [[0.0, 0.0, 3.6], [0.0, 3.6, 0.0], [3.6, 0.0, 0.0]]
    :return: frac_coord
    """
    a1 = np.array(basis[0])
    a2 = np.array(basis[1])
    a3 = np.array(basis[2])
    cart_coord = np.array(cart_coord)
    trans_matr = np.vstack([a1, a2, a3]).T
    inv_matr = np.linalg.inv(trans_matr)
    frac_coord = np.matmul(inv_matr, cart_coord.T).T

    return list(frac_coord)


def apply_pbc(clust, str_dict):
    """
    Apply periodic boundary conditions to the sites of a given cluster
    :param clust: containing scaled Cartesian coordinates
    :param str_dict: containing Cartesian coordinates
    :return: frac_clust
    """
    size = len(clust[0])
    pbc_clust = copy.deepcopy(clust)
    for i in range(size):
        pbc_clust[0][i] = cart_to_frac(clust[0][i], str_dict['LatVec'])
        pbc_clust[0][i] = np.around(pbc_clust[0][i], decimals=3)
        for j in range(3):
            pbc_clust[0][i][j] = pbc_clust[0][i][j] % 1
        pbc_clust[0][i] = frac_to_cart(pbc_clust[0][i], str_dict['LatVec'])

    return pbc_clust


def find_spec(clust, str_dict):
    """
    find the species on each sites of a given cluster
    :param clust: containing Cartesian coordinate after applying PBCs
    :param str_dict: containing Cartesian coordinates
    :return: spec_list: a list of species in the same sequence of the cluster sites
    """
    spec = [None] * len(clust[0])
    for i in range(len(clust[0])):
        dist = 0.1
        for j in range(len(str_dict['LatPnt'])):
            new_dist = calc_dist(clust[0][i], str_dict['LatPnt'][j])
            if new_dist < dist:
                dist = new_dist
                spec[i] = str_dict['Spec'][j]
    if None in spec:
        spec_list = ['empty']
    else:
        spec_list = copy.deepcopy(spec)

    return spec_list


def find_spin(clust, str_dict):
    """
    find the species on each sites of a given cluster
    :param clust: containing Cartesian coordinate after applying PBCs
    :param str_dict: containing Cartesian coordinates
    :return: spin product of all cluster sites
    """
    spin = np.zeros(len(clust[0]))
    for i in range(len(clust[0])):
        dist = 0.1
        for j in range(len(str_dict['LatPnt'])):
            if calc_dist(clust[0][i], str_dict['LatPnt'][j]) < dist:
                dist = calc_dist(clust[0][i], str_dict['LatPnt'][j])
                spin[i] = str_dict['Spin'][j]
    if None in spin:
        spin_value = None
    else:
        spin_value = math.prod(spin)

    return spin_value


def count_singlelattice(symeq_clust_list, pntsym_list, str_list, clust_list, spec_seq):
    """
    count the number of each cluster for each structure with single lattice
    :param symeq_clust_list: list of symmetry operation based on the input lattice file defined like ATAT
    :param pntsym_list: list of point symmetry operation for each symmetry equivalent cluster
    :param str_list: parsed DFT data list
    :param clust_list: parsed cluster list
    :param spec_seq: species order like ['Fe', 'Ni', 'Cr']
    :return: list of the count number (count_list)
    """
    count_list_all = []
    str_list = apply_basis(str_list)  # transform str from direct coord to Cartesian coord
    for str_dict in str_list:
        count_list = copy.deepcopy(clust_list)
        for i in range(len(clust_list)):
            count_dict = {}
            orig_clust = scale_clust(clust_list[i])  # transform clust from direct coord to Cartesian coord
            pbc_clust = apply_pbc(orig_clust, str_dict)
            spec = find_spec(pbc_clust, str_dict)
            if spec == ['empty']:
                print('Cluster #', i + 1, 'is not present in Structure #', str_dict['CellName'])
            else:
                symeq_clust = symeq_clust_list[i]
                multiplicity = len(symeq_clust)
                count_list[i].append({'Multiplicity': int(multiplicity/len(orig_clust[0]))})
                for j in range(len(str_dict['LatPnt'])):
                    for k in range(multiplicity):
                        old_clust = symeq_clust[k]
                        vect = np.subtract.reduce([str_dict['LatPnt'][j], [0, 0, 0]], axis=0)
                        new_clust = copy.deepcopy(old_clust)
                        for x in range(len(old_clust[0])):
                            new_clust[0][x] = np.sum([old_clust[0][x], vect], axis=0)
                        pbc_clust = apply_pbc(new_clust, str_dict)
                        spec = find_spec(pbc_clust, str_dict)
                        # find the only true equivalent sequence
                        spec = symop.find_eq_spec_seq(list(spec), old_clust, pntsym_list[i][k], spec_seq)
                        if new_clust[2][0] == 0:  # chem term
                            if str(spec) in count_dict.keys():
                                count_dict[str(spec)] += 1
                            else:
                                count_dict[str(spec)] = 1
                        elif new_clust[2][0] == 1:  # spin term
                            spin = find_spin(new_clust, str_dict)
                            if str(spec) in count_dict.keys():
                                count_dict[str(spec)] += spin
                            else:
                                count_dict[str(spec)] = spin
            for keys in count_dict:
                values = count_dict[keys]
                count_dict[keys] = np.around(values/(str_dict['AtomSum']*len(orig_clust[0])), decimals=5)
            count_list[i].append(count_dict)
        count_list_all.append([str_dict['CellName'], count_list])

    return count_list_all


def count_multisublattice():
    """
    count the number of each cluster for each structure with multisublattice
    :return:
    """
    return 0
