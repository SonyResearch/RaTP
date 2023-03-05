gpu_id=5

# --------------------------- main paper order ----------------------------- #
python main.py --gpu $gpu_id --dataset subdomain_net --order 3 5 0 1 2 4 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_mainpaper

python main.py --gpu $gpu_id --dataset subdomain_net --order 3 5 0 1 2 4 --seed 2023 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_mainpaper

python main.py --gpu $gpu_id --dataset subdomain_net --order 3 5 0 1 2 4 --seed 2024 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_mainpaper


# --------------------------- additional order ----------------------------- #
python main.py --gpu $gpu_id --dataset subdomain_net --order 4 2 1 0 5 3 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order

python main.py --gpu $gpu_id --dataset subdomain_net --order 0 1 2 3 4 5 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order

python main.py --gpu $gpu_id --dataset subdomain_net --order 5 4 3 2 1 0 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order

python main.py --gpu $gpu_id --dataset subdomain_net --order 2 5 3 1 4 0 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order

python main.py --gpu $gpu_id --dataset subdomain_net --order 0 4 1 3 5 2 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order

python main.py --gpu $gpu_id --dataset subdomain_net --order 3 4 0 2 1 5 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order

python main.py --gpu $gpu_id --dataset subdomain_net --order 5 1 2 0 4 3 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order

python main.py --gpu $gpu_id --dataset subdomain_net --order 1 3 0 2 4 5 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order

python main.py --gpu $gpu_id --dataset subdomain_net --order 5 4 2 0 3 1 --seed 2022 \
--aug_tau 0.8 --topk_alpha 10 --topk_beta 1 --lr 0.005 --MPCL_alpha 1.0 \
--output result_additional_order
