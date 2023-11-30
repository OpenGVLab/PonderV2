"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from ponder.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from ponder.engines.launch import launch, slurm_launch
from ponder.engines.train import TRAINERS


def main_worker(cfg):
    cfg = default_setup(cfg)
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    if args.launcher == "pytorch":
        launcher = launch
    elif args.launcher == "slurm":
        launcher = slurm_launch

    launcher(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        port=args.master_port,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()
