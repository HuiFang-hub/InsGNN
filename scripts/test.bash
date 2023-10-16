#!/bin/bash
# bash scripts/test.bash
#data_types=('ba_2motifs' "mutag" "cora" "spmotif_0.5" 'spmotif_0.7' 'spmotif_0.9')  #
#seeds=(1 2 3)
#loader_sizes=(32 64 128)
#graph_lrs=(0.01 0.001 0.0001)
#normal_coefs=(0 0.0001)
#kl_1_coefs=(0 0.0001 0.001 0.1 1)
#kl_2_coefs=(0 0.0001 0.001 0.1 1)
#GC_delta_coefs=(0 1)
#for data_type in "${data_types[@]}"; do
#   for seed in "${seeds[@]}"; do
#     for loader_size in "${loader_sizes[@]}"; do
#       for graph_lr in "${graph_lrs[@]}"; do
#         for normal_coef in "${normal_coefs[@]}"; do
#           for kl_1_coef in "${kl_1_coefs[@]}"; do
#             for kl_2_coef in "${kl_2_coefs[@]}"; do
#                 for GC_delta_coef in "${GC_delta_coefs[@]}"; do
#
#      python ./main.py  --device 2 --seed $seed --data['data_name'] $data_type  --data['loader_size'] $loader_size --train['graph_lr'] $graph_lr
       #      --model['normal_coef'] $normal_coef --model['kl_1_coef'] $kl_1_coef --model['kl_2_coef'] $kl_2_coef --model['GC_delta_coef'] $GC_delta_coef
#done
#done
#done
#done
#done


#data_types=('mutag')  #
#seeds=(1 2 3)
#loader_sizes=(128)
#graph_lrs=(0.01 0.001 0.0001)
#normal_coefs=(0.0 0.0001)
#kl_1_coefs=(0.0 0.0001 0.001 0.1 1.0)
#kl_2_coefs=(0.0 0.0001 0.001 0.1 1.0)
#GC_delta_coefs=(0.0 1.0)
#for seed in "${seeds[@]}"; do
#for data_type in "${data_types[@]}"; do
#     for loader_size in "${loader_sizes[@]}"; do
#       for graph_lr in "${graph_lrs[@]}"; do
#         for normal_coef in "${normal_coefs[@]}"; do
#           for kl_1_coef in "${kl_1_coefs[@]}"; do
#             for kl_2_coef in "${kl_2_coefs[@]}"; do
#                 for GC_delta_coef in "${GC_delta_coefs[@]}"; do
#      python main.py --cfg configs/test.yaml  device 2  seed $seed  data.data_name $data_type train.epochs 150  data.loader_size $loader_size train.graph_lr $graph_lr model.normal_coef $normal_coef  model.kl_1_coef $kl_1_coef  model.kl_2_coef $kl_2_coef  model.GC_delta_coef $GC_delta_coef
#done
#done
#done
#done
#done
#done
#done
#done

data_types=('ogbg_molhiv')  #
frameworks=('InsGNN')
seeds=(2 3)
normal_coefs=(0.0 )
kl_1_coefs=(1.0)
kl_2_coefs=(0.0)
GC_delta_coefs=(0.001)
for framework in "${frameworks[@]}"; do
for seed in "${seeds[@]}"; do
for data_type in "${data_types[@]}"; do
         for normal_coef in "${normal_coefs[@]}"; do
           for kl_1_coef in "${kl_1_coefs[@]}"; do
             for kl_2_coef in "${kl_2_coefs[@]}"; do
                 for GC_delta_coef in "${GC_delta_coefs[@]}"; do

      python main.py --cfg configs/GIN-ogbg_molhiv.yaml  device 0  seed $seed  framework $framework data.data_name $data_type train.epochs 150  model.normal_coef $normal_coef  model.kl_1_coef $kl_1_coef  model.kl_2_coef $kl_2_coef  model.GC_delta_coef $GC_delta_coef
done
done
done
done
done
done
done