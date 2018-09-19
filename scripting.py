import argparse
import logging
import os

def task_runner(tasks):
    LOG_LEVELS = {
        "DEBUG": logging.DEBUG, "INFO": logging.INFO, "ERROR": logging.ERROR
    }
    # parse cmd line arguments
    parser = argparse.ArgumentParser(description="run tasks")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-t", "--task", help="task to run", choices=tasks.keys()
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
    args = parser.parse_args()
    if args.task:
        # set logging
        logging.basicConfig(
            filename=os.path.join(args.log_dir, f"{args.task}.log"),
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=LOG_LEVELS[args.log_level]
        )
        # run chosen task
        task = tasks[args.task]
        task.run()
        return
    elif args.all:
        # set logging
        logging.basicConfig(
            filename=os.path.join(args.log_dir, f"all_tasks.log"),
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=LOG_LEVELS[args.log_level]
        )
        # run all tasks
        for task_name, task in tasks.items():
            logging.info(f"Running {task_name}")
            task.run()
        return
    else:
        print("Nothing to run")
        return
