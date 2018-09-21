import argparse
import logging
import os
import sys

def configure_logging(log_level, log_dir, task_name, batch):
    batch_str = "full_batch" if batch is None else str(batch)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"{task_name}.{batch_str}.log"),
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=log_level
    )

def run_task(task_name, task, batch, n_batch):
    logging.info(f"Running task {task_name}")
    if batch is None:
        logging.info(f"Running full batch")
        task.run()
    else:
        logging.info(f"Running batch {batch}/{n_batch}")
        task.run_batch(batch, n_batch)


def task_runner(tasks):
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG, "INFO": logging.INFO, "ERROR": logging.ERROR
    }
    # parse cmd line arguments
    parser = argparse.ArgumentParser(description="run tasks")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task", help="task to run", choices=tasks.keys()
    )
    group.add_argument(
        "-a", "--all", help="run all tasks",action="store_true"
    )
    parser.add_argument(
        "--log_level", default="DEBUG", help="logging level",
        choices=LOG_LEVELS.keys()
    )
    parser.add_argument(
        "--log_dir", default="logs", help="logs directory"
    )
    parser.add_argument(
        "--batch", type=int, help="batch in 1..n_batch"
    )
    parser.add_argument(
        "--n_batch", type=int, help="number of batches"
    )
    args = parser.parse_args()
    log_level = LOG_LEVELS[args.log_level]
    log_dir, task_name = args.log_dir, args.task
    batch, n_batch = args.batch, args.n_batch
    # check that batch arguments are consistent
    if batch is not None:
        if n_batch is None:
            sys.exit("you must provide n_batch")
        if (batch not in range(1, n_batch+1)):
            sys.exit("batch must be in 1..n_batch")
    if task_name:
        configure_logging(log_level, log_dir, task_name, batch)
        # run chosen task
        task = tasks[task_name]
        run_task(task_name, task, batch, n_batch)
        return
    elif args.all:
        configure_logging(log_level, log_dir, "all_tasks", batch)
        # run all tasks
        for task_name, task in tasks.items():
            run_task(task_name, task, batch, n_batch)
        return
    else:
        print("Nothing to run")
        return
