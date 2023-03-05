gpu_id=2

# --------------------------- main paper order ----------------------------- #
python main.py --gpu $gpu_id --order 2 0 1 3 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_mainpaper 

python main.py --gpu $gpu_id --order 2 0 1 3 --seed 2023 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_mainpaper 

python main.py --gpu $gpu_id --order 2 0 1 3 --seed 2024 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_mainpaper 


python main.py --gpu $gpu_id --order 3 1 0 2 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_mainpaper 

python main.py --gpu $gpu_id --order 3 1 0 2 --seed 2023 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_mainpaper 

python main.py --gpu $gpu_id --order 3 1 0 2 --seed 2024 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_mainpaper 


# --------------------------- additional order ----------------------------- #

python main.py --gpu $gpu_id --order 0 1 2 3 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_additional_order 

python main.py --gpu $gpu_id --order 0 1 3 2 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_additional_order 

python main.py --gpu $gpu_id --order 0 2 1 3 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_additional_order 

python main.py --gpu $gpu_id --order 1 0 3 2 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_additional_order 

python main.py --gpu $gpu_id --order 1 3 2 0 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_additional_order 

python main.py --gpu $gpu_id --order 2 3 0 1 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_additional_order 

python main.py --gpu $gpu_id --order 2 3 1 0 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_additional_order 

python main.py --gpu $gpu_id --order 3 2 1 0 --seed 2022 \
--aug_tau 0.5 --topk_alpha 20 --lr 0.005 --MPCL_alpha 0.5 \
--output result_additional_order 