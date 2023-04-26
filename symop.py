import copy
import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.symmetry import analyzer as syman
from pymatgen.symmetry import site_symmetries as symsite


def war_pmg_sym(pmg_sym):
    """
    write and read pymatgen symmetry
    :param pmg_sym: pymatgen symmetry operations
    :return: list of symmetry operations from pmg_sym
    """
    f = open('sym_temp', 'w+')
    f.write(str(pmg_sym))
    f.close()
    lines = open('sym_temp', 'r').read().split('\n')
    sym_list = []
    for i in range(len(lines)):
        if ':' in lines[i]:
            s = []
            for j in range(1, 4):
                a = lines[i + j].replace('[', '')
                b = a.replace(']', '')
                c = b.split()
                d = [round(float(c[0]), 5), round(float(c[1]), 5), round(float(c[2]), 5)]
                s.append(d)
            sym_list.append(s)

    return sym_list


def find_sym(lat_in):
    """
    find the site symmetry of a given lattice
    :param lat_in: lattice file
    :return: list of symmetry operations
    """
    struct = Structure.from_file(lat_in)
    struct = syman.SpacegroupAnalyzer(struct).find_primitive()
    pmg_sym = symsite.get_site_symmetries(struct, 0.1)

    return war_pmg_sym(pmg_sym)


def calc_dist(pnt1, pnt2):
    """
    calculate the distance between two point in 3D Cartesian coordinate
    :param pnt1: list
    :param pnt2: list
    :return: dist
    """
    dist = np.linalg.norm(np.array(pnt1) - np.array(pnt2))

    return dist


def check_lateral(clust):
    """
    if the given cluster is equilateral, return 1
    :param clust: a cluster from a cluster list
    :return: boolean
    """
    if len(clust[0]) > 2:
        center_coord = list(np.mean(clust[0], axis=0))
        dist = calc_dist(center_coord, clust[0][0])
        for i in range(len(clust[0])):
            if np.abs(dist - calc_dist(center_coord, clust[0][i])) > 0.001:
                return 0
    return 1


def check_uniq(clust, clust_list):
    """
    if a cluster is not present in a cluster list allowing site permutation, return 1
    note that by applying this function, every cluster is a sorted list in terms of cluster site coordinates
    :param clust: a cluster
    :param clust_list: a cluster list
    :return: boolean
    """
    clust[0].sort()
    for uniq_clust in clust_list:
        uniq_clust[0].sort()
        if uniq_clust[0] == clust[0]:
            return 0
    return 1


def sort_clust_dist(clust):
    """
    for a cluster that is not equilateral, sort the site coordinates in a specific sequence
    :param clust: scaled and symmetry-applied cluster in Cartesian coordinate
    :return: cluster with sorted sites
    """
    size = len(clust[0])
    dist_matr = np.zeros((size, size))
    dist_list = [[]] * size
    for i in range(size):
        for j in range(1, size):
            dist_matr[i][j] = calc_dist(clust[0][i], clust[0][j])
            dist_matr[j][i] = dist_matr[i][j]
        dist_list[i] = list(dist_matr[i])
        dist_list[i].pop(i)
        dist_list[i].sort()
        dist_list[i].append(clust[0][i])
    dist_list.sort()
    for i in range(size):
        clust[0][i] = dist_list[i][size-1]

    return clust


def sort_coord_dist(coords):
    """
    sort the site coordinates and species in a specific sequence
    e.g.sort the list of [[0.0, 0.0, 0.0, 'Fe'], [1.8, 1.8, 0.0, 'Fe'], [-1.8, 1.8, 0.0, 'Ni']]
    to  [[-1.8, 1.8, 0.0, 'Ni'], [0.0, 0.0, 0.0, 'Fe'], [1.8, 1.8, 0.0, 'Fe']]
    :param coords: scaled and symmetry-applied cluster in Cartesian coordinate
    :return: coordinates of sorted sites
    """
    size = len(coords)
    dist_matr = np.zeros((size, size))
    dist_list = [[]] * size
    site = copy.deepcopy(coords)
    for i in range(size):
        site[i].pop(3)
    for i in range(size):
        for j in range(1, size):
            dist_matr[i][j] = calc_dist(site[i], site[j])
            dist_matr[j][i] = dist_matr[i][j]
        dist_list[i] = list(dist_matr[i])
        dist_list[i].pop(i)
        dist_list[i].sort()
        dist_list[i].append(coords[i])
    dist_list.sort()
    for i in range(size):
        coords[i] = dist_list[i][size-1]

    return coords


def apply_symop(sym_op, clust):
    """
    apply a symmetry operation to a cluster with one site at origin [0, 0, 0]
    :param sym_op: symmetry operation
    :param clust: cluster from scaled clust_in file in Cartesian coordinate
    :return: cluster after symmetry operation
    """
    symeq_clust = copy.deepcopy(clust)
    for j in range(len(clust[0])):
        clust_site = clust[0][j]
        symeq_clust_site = np.matmul(sym_op, np.array(clust_site).T).T
        symeq_clust[0][j] = list(np.around(symeq_clust_site, decimals=3))
    symeq_clust[0].sort(key=lambda x: (x[0], x[1], x[2]))

    return symeq_clust


def find_eq_clust(sym_list, clust):
    """
    find all symmetry-equivalent clusters allowing site permutation for a given cluster
    note that by applying this function, every cluster is a sorted list in terms of cluster site coordinates
    :param sym_list: list of symmetry operations
    :param clust: cluster from scaled clust_in file in Cartesian coordinate
    :return: list of symmetry-equivalent clusters of the given cluster
    """
    symeq_clust_list = []
    if check_lateral(clust):
        vect = np.subtract.reduce([[0, 0, 0], clust[0][0]], axis=0)
        for x in range(len(clust[0])):
            clust[0][x] = np.sum([clust[0][x], vect], axis=0)
        for i in range(len(sym_list)):
            sym_op = np.vstack(sym_list[i])
            symeq_clust = apply_symop(sym_op, clust)
            if check_uniq(symeq_clust, symeq_clust_list):
                symeq_clust_list.append(symeq_clust)
        return symeq_clust_list
    else:
        for j in range(len(clust[0])):
            vect = np.subtract.reduce([[0, 0, 0], clust[0][j]], axis=0)
            for x in range(len(clust[0])):
                clust[0][x] = np.sum([clust[0][x], vect], axis=0)
            for i in range(len(sym_list)):
                sym_op = np.vstack(sym_list[i])
                symeq_clust = apply_symop(sym_op, clust)
                symeq_clust[0].sort()
                symeq_clust_list.append(symeq_clust)
                # if check_uniq(symeq_clust, symeq_clust_list):
                #     symeq_clust_list.append(symeq_clust)
        symeq_clust_list = [x for n, x in enumerate(symeq_clust_list) if x not in symeq_clust_list[:n]]
        for clust in symeq_clust_list:
            sort_clust_dist(clust)

        return symeq_clust_list


def find_eq_spec_seq(spec, clust, pntsym):
    """
    find all equivalent species sequences in a given cluster
    :param spec: input species sequence
    :param clust: site-sorted cluster from symeq_clust_list
    :return: the first item in the sorted list of equivalent species sequences
    """
    if len(spec) == 1:
        return spec
    elif len(spec) == 2:
        if spec[0] == spec[1]:
            return spec
        else:
            spec.sort()
            return spec
    elif len(spec) >= 3:
        coords = clust[0]
        # hypo_molec = Molecule(['H'] * len(coords), coords)
        # pntsym = syman.PointGroupAnalyzer(hypo_molec).get_symmetry_operations()
        new_spec_list = []
        new_clust_list = []
        for sym_op in pntsym:
            new_clust = [[]] * len(spec)
            molec = Molecule(spec, coords)
            molec.apply_operation(sym_op)
            for i in range(len(spec)):
                pnt = str(molec[i])
                a = pnt.replace('[', '')
                b = a.replace(']', '')
                c = b.split()
                d = [round(float(c[0]), 5), round(float(c[1]), 5), round(float(c[2]), 5), str(c[3])]
                new_clust[i] = d
            new_clust.sort()
            new_clust_list.append(new_clust)
        new_clust_list = [x for n, x in enumerate(new_clust_list) if x not in new_clust_list[:n]]
        for i in range(len(new_clust_list)):
            new_spec = []
            new_clust = new_clust_list[i]
            sort_coord_dist(new_clust)
            for i in range(len(new_clust)):
                new_spec.append(new_clust[i][3])
            new_spec_list.append(new_spec)
        new_spec_list.sort()

        return new_spec_list[0]


# sp = [[]] * 3
# cl = [[]] * 3
# el = [[]] * 3
# sp[0] = ['1', '1', '0', '1']
# el[0] = ['Ni', 'Cr', 'Fe', 'Ni']
# cl[0] = [[[0, 0, 0], [1.8, 1.8, 0], [0, 1.8, 0], [1.8, 0, 0]], [1.8], [0]]
# sp[1] = ['2', '1']
# cl[1] = [[[0, 0, 0], [1, 0, 0]], [1], [0]]
# sp[2] = ['1', '1', '0']
# cl[2] = [[[-1.8, -1.8, 0], [-3.6, 0, 0], [0, 0, 0]], [3.6], [0]]
#
# y = find_eq_spec_seq(el[0],cl[0])
# print(y, len(y))

# sym_list = find_sym('lat.in')
# x = find_eq_clust(sym_list, cl[2])
# print(x, len(x))


# cl1 = [[[-1.8, -1.8, 0], [-3.6, 0, 0], [0, 0, 0]], [3.6], [0]]
# cl2 = [[[1.8, -1.8, 0], [0, 0, 0], [3.6, 0, 0]], [3.6], [0]]
# cl3 = [[[0, 0, 0], [1.8, -1.8, 0], [1.8, 1.8, 0], [3.6, 0, 0]], [3.6], [0]]
# cl4 = [[[0, 0, 0], [0, 1.8, 0], [1.8, 0, 0], [1.8, 1.8, 0], [1.9, 0.9, 0]], [3.6], [0]]
# spec = ['1', '1', '0', '0']
# elem = ['Ni', 'Fe', 'Fe', 'Ni', 'Cr']
# coords = cl4[0]
# molec = Molecule(['H'] * len(coords), coords)
# pntsym = syman.PointGroupAnalyzer(molec).get_symmetry_operations()
# print(len(pntsym))
# clust = Molecule(elem, coords)
# new_clust = [[]] * len(elem)
# for sym_op in pntsym:
#     clust = Molecule(elem, coords)
#     clust.apply_operation(sym_op)
#     for i in range(len(clust)):
#         pnt = str(clust[i])
#         a = pnt.replace('[', '')
#         b = a.replace(']', '')
#         c = b.split()
#         d = [round(float(c[0]), 5), round(float(c[1]), 5), round(float(c[2]), 5), str(c[3])]
#         new_clust[i] = d
#     print('old', new_clust)
#     sort_coord_dist(new_clust)
#     print('new', new_clust)
