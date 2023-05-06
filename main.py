import parse
import count
import symop
import cefit
import numpy as np
from pymatgen.core import Molecule
from pymatgen.symmetry import analyzer as syman
import json
import time
start_time = time.time()


lat_in = 'lat.in'
data_file = 'full_data'
clust_in = 'clusters_FCC'
species = ['Fe', 'Ni', 'Cr']

if __name__ == '__main__':
    str_list = parse.parse_str(data_file)
    str_list = parse.find_uniq_str(str_list)
    print(len(str_list))
    with open('str_out', 'w') as filehandle:
        json.dump(str_list, filehandle)
    clust_list = parse.parse_clust(clust_in)
    sym_list = symop.find_sym(lat_in)
    symeq_clust_list = []
    new_clust_list = []
    for orig_clust in clust_list:
        orig_clust = count.scale_clust(orig_clust)
        symeq_clust_list.append(symop.find_eq_clust(sym_list, orig_clust))
    with open('symeq_clust_out', 'w') as filehandle:
        json.dump(symeq_clust_list, filehandle)
    spec_pntsym_list = []
    for symeq_clust in symeq_clust_list:
        new_clust_list.append(symeq_clust[0])
        if len(symeq_clust[0][0]) <= 2:
            spec_pntsym_list.append([[None]] * len(symeq_clust))
        else:
            pntsym_list = []
            for clust in symeq_clust:
                coords = clust[0]
                hypo_molec = Molecule(['H'] * len(coords), coords)
                pntsym = syman.PointGroupAnalyzer(hypo_molec).get_symmetry_operations()
                pntsym_list.append(pntsym)
            spec_pntsym_list.append(pntsym_list)
    count_list = count.count_singlelattice(symeq_clust_list, spec_pntsym_list, str_list, new_clust_list, species)
    print(len(count_list))
    with open('count_out', 'w') as filehandle:
        json.dump(count_list, filehandle)

    count = parse.parse_count('count_out')
    enrg = parse.parse_enrg('str_out')
    print(len(count), len(enrg))
    ridge_eci = cefit.ridge_fit(count, enrg)
    lasso_eci = cefit.lasso_fit(count, enrg)


print("--- %s seconds ---" % (time.time() - start_time))
