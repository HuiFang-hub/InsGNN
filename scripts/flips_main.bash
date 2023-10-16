#!/bin/bash
# bash scripts/flips_main.bash

#data_types=('spmotif_0.9')  #
#frameworks=('GAT' 'GSAT')
#rates=(0.2 0.4 0.6 0.8)
#normal_coefs=(0.0 )
#kl_1_coefs=(0.0)
#kl_2_coefs=(0.0)
#GC_delta_coefs=(0.0)
#for framework in "${frameworks[@]}"; do
#for r in "${rates[@]}"; do
#for data_type in "${data_types[@]}"; do
#         for normal_coef in "${normal_coefs[@]}"; do
#           for kl_1_coef in "${kl_1_coefs[@]}"; do
#             for kl_2_coef in "${kl_2_coefs[@]}"; do
#                 for GC_delta_coef in "${GC_delta_coefs[@]}"; do
#
#      python flips_main.py --cfg configs/flips_main.yaml  device 0  data.r $r  framework $framework data.data_name $data_type train.epochs 150  model.normal_coef $normal_coef  model.kl_1_coef $kl_1_coef  model.kl_2_coef $kl_2_coef  model.GC_delta_coef $GC_delta_coef
#done
#done
#done
#done
#done
#done
#done

data_types=('spmotif_0.9')  #
frameworks=('InsGNN')
rates=(0.2 0.4 0.6 0.8)
normal_coefs=(0.0)
kl_1_coefs=(0.001)
kl_2_coefs=(0.0)
GC_delta_coefs=(0.0)
for framework in "${frameworks[@]}"; do
for r in "${rates[@]}"; do
for data_type in "${data_types[@]}"; do
         for normal_coef in "${normal_coefs[@]}"; do
           for kl_1_coef in "${kl_1_coefs[@]}"; do
             for kl_2_coef in "${kl_2_coefs[@]}"; do
                 for GC_delta_coef in "${GC_delta_coefs[@]}"; do

      python flips_main.py --cfg configs/flips_main.yaml  device 0  data.r $r  framework $framework data.data_name $data_type train.epochs 150  model.normal_coef $normal_coef  model.kl_1_coef $kl_1_coef  model.kl_2_coef $kl_2_coef  model.GC_delta_coef $GC_delta_coef
done
done
done
done
done
done
done