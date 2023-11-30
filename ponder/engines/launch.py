"""
Launcher

modified from detectron2(https://github.com/facebookresearch/detectron2)

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import logging
import os
import subprocess
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ponder.utils import comm

__all__ = ["DEFAULT_TIMEOUT", "launch", "slurm_launch"]

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    port=None,
    cfg=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.
    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        if dist_url == "auto":
            assert (
                num_machines == 1
            ), "dist_url=auto not supported in multi-machine jobs."
            port = _find_free_port()
            dist_url = f"tcp://127.0.0.1:{port}"
        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        mp.spawn(
            _distributed_worker,
            nprocs=num_gpus_per_machine,
            args=(
                main_func,
                world_size,
                num_gpus_per_machine,
                machine_rank,
                dist_url,
                cfg,
                timeout,
            ),
            daemon=False,
        )
    else:
        main_func(*cfg)


def _distributed_worker(
    local_rank,
    main_func,
    world_size,
    num_gpus_per_machine,
    machine_rank,
    dist_url,
    cfg,
    timeout=DEFAULT_TIMEOUT,
):
    assert (
        torch.cuda.is_available()
    ), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
            timeout=timeout,
        )
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error("Process group URL: {}".format(dist_url))
        raise e

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(
            range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
        )
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # synchronize is needed here to prevent a possible timeout after calling init_process_group
    # See: https://github.com/facebookresearch/maskrcnn-benchmark/issues/172
    comm.synchronize()

    main_func(*cfg)


def slurm_launch(
    main_func,
    num_gpus_per_machine,
    num_machines=1,
    machine_rank=0,
    dist_url=None,
    port=None,
    backend="nccl",
    cfg=(),
    timeout=DEFAULT_TIMEOUT,
):
    """
    Launch multi-gpu or distributed training.
    This function must be called on all machines involved in the training.
    It will spawn child processes (defined by ``num_gpus_per_machine``) on each machine.

    Args:
        main_func: a function that will be called by `main_func(*args)`
        num_gpus_per_machine (int): number of GPUs per machine
        num_machines (int): the total number of machines
        machine_rank (int): the rank of this machine
        dist_url (str): url to connect to for distributed jobs, including protocol
                       e.g. "tcp://127.0.0.1:8686".
                       Can be set to "auto" to automatically select a free port on localhost
        timeout (timedelta): timeout of the distributed workers
        args (tuple): arguments passed to main_func
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method("spawn")
    world_size = num_machines * num_gpus_per_machine
    if world_size > 1:
        # https://github.com/pytorch/pytorch/pull/14391
        # TODO prctl in spawned processes

        if num_machines > 1 and dist_url.startswith("file://"):
            logger = logging.getLogger(__name__)
            logger.warning(
                "file:// is not a reliable init_method in multi-machine jobs. Prefer tcp://"
            )

        """Initialize slurm distributed training environment.

        If argument ``port`` is not specified, then the master port will be system
        environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
        environment variable, then a default port ``29500`` will be used.

        Args:
            backend (str): Backend of torch.distributed.
            port (int, optional): Master port. Defaults to None.
        """
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        local_rank_env = os.environ.get("SLURM_LOCALID", None)
        if local_rank_env is not None:
            local_rank = int(local_rank_env)
        else:
            num_gpus = torch.cuda.device_count()
            local_rank = proc_id % num_gpus
        torch.cuda.set_device(local_rank)
        machine_rank = proc_id // num_gpus
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")

        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            # 29500 is torch.distributed default port
            os.environ["MASTER_PORT"] = "29500"
            # if dist_url.startswith("tcp://"):
            #     port = dist_url.split(":")[-1]
            #     print("dist_url: ", dist_url, " port: ", port)
            # os.environ["MASTER_PORT"] = port

        # use MASTER_ADDR in the environment variable if it already exists
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["LOCAL_RANK"] = str(local_rank)
        os.environ["RANK"] = str(proc_id)
        dist.init_process_group(backend=backend, timeout=timeout)
        comm.synchronize()

        assert comm._LOCAL_PROCESS_GROUP is None
        num_machines = world_size // num_gpus_per_machine
        for i in range(num_machines):
            ranks_on_i = list(
                range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine)
            )
            pg = dist.new_group(ranks_on_i)
            if i == machine_rank:
                comm._LOCAL_PROCESS_GROUP = pg

        main_func(*cfg)
    else:
        main_func(*cfg)
