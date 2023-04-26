import parse
import count
import symop
import cefit
import numpy as np
from pymatgen.core import Structure, Molecule
from pymatgen.symmetry import analyzer as syman
from pymatgen.symmetry import site_symmetries as symsite
import time
start_time = time.time()


def write_count(count_list):
    return count_list


lat_in = 'lat.in'
data_file = 'E_Mag_output'
clust_in = 'clusters_FCC'

if __name__ == '__main__':
    str_list = parse.parse_str(data_file)
    clust_list = parse.parse_clust(clust_in)
    sym_list = symop.find_sym(lat_in)

    # for clust in clust_list:
    #     sym_eq_list = symop.find_eq_clust(sym_list, clust)
    #     print(len(sym_eq_list), sym_eq_list)
    # a = symop.find_eq_clust(sym_list, clust_list[3])
    # print(len(a), a)
    symeq_clust_list = []
    for orig_clust in clust_list:
        orig_clust = count.scale_clust(orig_clust)
        symeq_clust_list.append(symop.find_eq_clust(sym_list, orig_clust))
    spec_pntsym_list = []
    for symeq_clust in symeq_clust_list:
        if len(symeq_clust[0][0]) <=2:
            spec_pntsym_list.append([[None]] * len(symeq_clust))
        else:
            pntsym_list = []
            for clust in symeq_clust:
                coords = clust[0]
                hypo_molec = Molecule(['H'] * len(coords), coords)
                pntsym = syman.PointGroupAnalyzer(hypo_molec).get_symmetry_operations()
                pntsym_list.append(pntsym)
            spec_pntsym_list.append(pntsym_list)
    count_list = count.count_singlelattice(symeq_clust_list, spec_pntsym_list, str_list, clust_list)
    print(count_list)

print("--- %s seconds ---" % (time.time() - start_time))
