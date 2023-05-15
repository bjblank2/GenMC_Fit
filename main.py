import parse
import count
import symop
import cefit
from pymatgen.core import Molecule
from pymatgen.symmetry import analyzer as syman
import json
import time
start_time = time.time()

do_count = True
do_fit = True
lat_in = 'lat.in'
data_file = 'test_FeNiCr'
clust_in = 'clusters_FCC'
species = ['Fe', 'Ni', 'Cr']


str_list = parse.parse_str(data_file)
str_list = parse.find_uniq_str(str_list)
print('# of unique structures', len(str_list))
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

if do_count:
    count_list = count.count_singlelattice(symeq_clust_list, spec_pntsym_list, str_list, new_clust_list, species)
    with open('count_out', 'w') as filehandle:
        json.dump(count_list, filehandle)

if do_fit:
    count, deco_list = parse.parse_count('count_out')
    with open('deco_out', 'w') as filehandle:
        json.dump(deco_list, filehandle)
    enrg = parse.parse_enrg('str_out')
    # ridge_eci = cefit.ridge_fit(count, enrg)
    # with open('ridge_eci', 'w') as filehandle:
    #     json.dump(ridge_eci.tolist(), filehandle)
    lasso_eci = cefit.lasso_fit(count, enrg)
    with open('lasso_eci', 'w') as filehandle:
        json.dump(lasso_eci.tolist(), filehandle)
    # eln_eci = cefit.eln_fit(count, enrg)
    # with open('eln_eci', 'w') as filehandle:
    #     json.dump(eln_eci.tolist(), filehandle)

    # write MC rules after fitting
with open('symeq_clust_out', 'r') as filehandle:
    symeq_clust_list = json.load(filehandle)
with open('deco_out', 'r') as filehandle:
    deco_list = json.load(filehandle)
with open('lasso_eci', 'r') as filehandle:
    eci_list = json.load(filehandle)
cefit.write_eci(symeq_clust_list, deco_list, eci_list, spec_pntsym_list, species)

print("--- %s seconds ---" % (time.time() - start_time))
