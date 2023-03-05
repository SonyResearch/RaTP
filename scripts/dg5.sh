gpu_id=3

# --------------------------- main paper order ----------------------------- #
python main.py --gpu $gpu_id --dataset dg5 --order 0 1 2 3 4 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_mainpaper

python main.py --gpu $gpu_id --dataset dg5 --order 0 1 2 3 4 --net DTN --seed 2023 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_mainpaper

python main.py --gpu $gpu_id --dataset dg5 --order 0 1 2 3 4 --net DTN --seed 2024 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_mainpaper


python main.py --gpu $gpu_id --dataset dg5 --order 4 3 2 1 0 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_mainpaper

python main.py --gpu $gpu_id --dataset dg5 --order 4 3 2 1 0 --net DTN --seed 2023 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_mainpaper

python main.py --gpu $gpu_id --dataset dg5 --order 4 3 2 1 0 --net DTN --seed 2024 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_mainpaper


# --------------------------- additional order ----------------------------- #
python main.py --gpu $gpu_id --dataset dg5 --order 0 1 4 2 3 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_additional_order 

python main.py --gpu $gpu_id --dataset dg5 --order 1 4 0 3 2 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_additional_order 

python main.py --gpu $gpu_id --dataset dg5 --order 2 0 1 3 4 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_additional_order 

python main.py --gpu $gpu_id --dataset dg5 --order 2 3 0 4 1 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_additional_order 

python main.py --gpu $gpu_id --dataset dg5 --order 3 1 2 0 4 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_additional_order 

python main.py --gpu $gpu_id --dataset dg5 --order 3 2 4 1 0 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_additional_order 

python main.py --gpu $gpu_id --dataset dg5 --order 3 4 1 2 0 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_additional_order 

python main.py --gpu $gpu_id --dataset dg5 --order 4 0 2 1 3 --net DTN --seed 2022 \
--aug_tau 0.8 --topk_alpha 20 --pseudo_fre 2 --lr 0.01 --MPCL_alpha 1 \
--output result_additional_order 
