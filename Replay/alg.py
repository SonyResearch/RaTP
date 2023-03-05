# coding=utf-8
from Replay.iCaRL import iCaRL
from Replay.Finetune import Finetune
# from Replay.LDAuCID_buff import LDAuCID_buff

ALGORITHMS = [
    'iCaRL',
    'Finetune'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
