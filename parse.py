import json
import numpy as np


def parse_str(data_file):
    """
    Read in the DFT data and return to a list containing metadata dictionary for all DFT calculations
    :param data_file: file created by the vasp_compilation code from DFT data set
    :return: parsed DFT data list (str_list)
    """
    f = open(data_file)  # Read in DFT data
    data = f.readlines()
    f.close()
    str_dict = {}  # Dictionary containing all data and metadata for each DFT calculation
    str_list = []
    for i in range(len(data)):  # Begin parsing DFT data file
        if "#" in data[i]:  # "#" indicates a new DFT data point
            elem_name = data[i].split()  # List of the chemical species in the DFT data
            elem_name.pop(0)
            elem_num = len(elem_name)
            set_data = data[i + 1]
            set_data = set_data.split()
            name = set_data[elem_num]
            atom_numb = [int(set_data[i]) for i in range(elem_num)]
            atom_sum = sum(atom_numb)  # Total number of atoms in each DFT data point
            enrg = float(set_data[elem_num + 1]) / atom_sum
            lat_const = [float(set_data[elem_num + 2]), float(set_data[elem_num + 3]),
                         float(set_data[elem_num + 4])]
            lat_ang = [float(set_data[elem_num + 5]), float(set_data[elem_num + 6]),
                       float(set_data[elem_num + 7])]
            lat_vec = [data[i + 2 + j].split() for j in range(3)]
            lat_vec = [[float(lat_vec[j][k]) for k in range(len(lat_vec[j]))] for j in range(len(lat_vec))]
            vol = lat_const[0] * lat_const[1] * lat_const[2] * np.sqrt(1 - np.power(
                np.cos(lat_ang[0]), 2) - np.power(np.cos(lat_ang[1]), 2) - np.power(np.cos(lat_ang[2]), 2)
                    + 2 * np.cos(lat_ang[0]) * np.cos(lat_ang[1]) * np.cos(lat_ang[2])) / atom_sum
            lat_type = 'Direct'
            pos_list = []
            spin_list = []
            type_list = []
            spec_list = []
            for j in range(int(atom_sum)):
                line = data[i + j + 5]
                line = line.split()
                spin = float(line[2])
                atom_type = line[1]
                line = [float(line[k]) for k in range(3, 6)]
                atom_pos = line  # np.dot(np.transpose(line), lat_vec)
                pos_list.append(atom_pos)
                spin_list.append(spin)
                type_list.append(atom_type)
            for j in range(len(elem_name)):
                for k in range(int(atom_numb[j])):
                    spec_list.append(elem_name[j])
            str_dict['CellName'] = name  # structure names
            str_dict['LatVec'] = lat_vec  # lattice vectors in 3D
            str_dict['LatConst'] = lat_const  # lattice constants
            str_dict['UnitVol'] = vol  # volume per cell
            str_dict['Enrg'] = enrg  # energy per cell
            str_dict['ElemName'] = elem_name  # species names
            str_dict['ElemNum'] = elem_num  # number of species type
            str_dict['AtomNum'] = atom_numb  # number of each atomic species
            str_dict['AtomSum'] = atom_sum  # total number of atom
            str_dict['LatType'] = lat_type  # use direct coordinate
            str_dict['LatPnt'] = pos_list  # list of coordinates for each atom
            str_dict['Spin'] = spin_list  # list of spin at each atom
            str_dict['Type'] = type_list  # list of species of each atom
            str_dict['Spec'] = spec_list
            str_list.append(str_dict.copy())
    return str_list


def parse_clust(clust_in):
    """
    Read in the cluster rules and return to the list of clusters
    :param clust_in: file containing cluster rules specified by users
    :return: parsed cluster list (clust_list)
    """
    with open(clust_in) as f:
        data_clust = json.load(f)
    clust_list = data_clust['List']  # list of clusters
    return clust_list
