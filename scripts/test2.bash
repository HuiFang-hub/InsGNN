#!/bin/bash
# bash scripts/test2.bash

data_types=('ogbg_molhiv')  #
frameworks=('InsGNN')
seeds=(1 2 3)
normal_coefs=(1.0 )
kl_1_coefs=(0.01)
kl_2_coefs=(0.0)
GC_delta_coefs=(0.1)
for framework in "${frameworks[@]}"; do
for seed in "${seeds[@]}"; do
for data_type in "${data_types[@]}"; do
         for normal_coef in "${normal_coefs[@]}"; do
           for kl_1_coef in "${kl_1_coefs[@]}"; do
             for kl_2_coef in "${kl_2_coefs[@]}"; do
                 for GC_delta_coef in "${GC_delta_coefs[@]}"; do

      python main.py --cfg configs/GIN-ogbg_molhiv.yaml  device 3  seed $seed  framework $framework data.data_name $data_type train.epochs 150  model.normal_coef $normal_coef  model.kl_1_coef $kl_1_coef  model.kl_2_coef $kl_2_coef  model.GC_delta_coef $GC_delta_coef
done
done
done
done
done
done
done
