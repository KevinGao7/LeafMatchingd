# #!/bin/bash

# Efficiency
for dataset in cora ogbl_collab ogbl_ppa icews18_min icews18_mid icews18_max ro internet kegg; do
        python undir_getdata.py --dataset $dataset --method normal 
done
python undir_getdata_sca.py &

