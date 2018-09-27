import pandas as pd
import numpy as np
import time
import logging
import itertools
from .connection import TableResultHandler


def log_on_progress(i, total):
    logging.info(f"experiment {i}/{total}")


class DataFrameResultHandler():
    def __init__(self):
        self._records = []

    def add_record(self, record):
        self._records.append(record)

    def get_dataframe(self):
        return pd.DataFrame(self._records)


def run_multiple_experiments(experiments, run, result_handler, on_progress=None):
    if not on_progress:
        on_progress = log_on_progress
    # iterate over experiments
    nb_experiments = len(experiments)
    for idx, experiment in enumerate(experiments):
        record = experiment.copy()
        try:
            result = run(**experiment)
            record.update(result)
            result_handler.add_record(record)
        except Exception:
            logging.exception(f"Experiment {experiment} failed")
        on_progress(idx + 1, nb_experiments)


def as_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, np.ndarray):
        return list(x)
    else:
        return [x]


def get_batch_positions(n_total, n_batch):
    """Return the batch positions when dividing n_total in n_batch."""
    if (n_batch < 1):
        raise ValueError("n_batch must be greater than 1")
    if (n_batch > n_total):
        raise ValueError("n_batch must be less than n_total")
    # n_total = q*n_batch + r = r*(q+1) + (n_batch-r)*q
    q, r = divmod(n_total, n_batch)
    # r batches of size q+1 and (n_batch-r) batches of size q
    batch_sizes = r * [q + 1] + (n_batch - r) * [q]
    batch_end, batch_positions = 0, []
    for batch_size in batch_sizes:
        batch_start = batch_end
        batch_end = batch_start + batch_size
        batch_positions.append((batch_start, batch_end))
    return batch_positions


class Task():
    def __init__(self, run_experiment, **kwargs):
        self.run_experiment = run_experiment
        self.kwargs = kwargs
        self.experiments = None

    def _experiments(self):
        kwargs = {key: as_list(val) for key, val in self.kwargs.items()}
        self.experiments = [
            {key: value for key, value in zip(kwargs.keys(), record_values)}
            for record_values in itertools.product(*kwargs.values())
        ]

    def _run(self, experiments, result_handler, on_progress):
        fname, nb = self.run_experiment.__name__, len(experiments)
        logging.info(f"Running {fname} on {nb} experiments")
        start = time.time()
        run_multiple_experiments(
            experiments, self.run_experiment, result_handler, on_progress
        )
        elapsed_time = time.time() - start
        logging.info(f"Task done in {elapsed_time:.1f}s")

    def run(self, result_handler, on_progress):
        if not self.experiments:
            self._experiments()
        self._run(self.experiments, result_handler, on_progress)

    def run_batch(self, batch, n_batch, result_handler, on_progress):
        if (batch not in range(1, n_batch+1)):
            raise ValueError(f"batch={batch} not in 1..{n_batch}")
        if not self.experiments:
            self._experiments()
        n_total = len(self.experiments)
        batch_positions = get_batch_positions(n_total, n_batch)
        batch_start, batch_end = batch_positions[batch-1]
        batch_experiments = self.experiments[batch_start:batch_end]
        logging.info(f"Experiments [{batch_start+1}..{batch_end}]/{n_total}")
        self._run(batch_experiments, result_handler, on_progress)


class TableTask(Task):
    def __init__(self, table_name, run_experiment, **kwargs):
        super().__init__(run_experiment, **kwargs)
        self.table_name = table_name

    def run(self, on_progress=None):
        result_handler = TableResultHandler(self.table_name)
        super().run(result_handler, on_progress)

    def run_batch(self, batch, n_batch, on_progress=None):
        result_handler = TableResultHandler(self.table_name)
        super().run_batch(batch, n_batch, result_handler, on_progress)


def teacher_student_scenario(n_samples, teacher, student, return_all=False, n_test=1000):
    """Run teacher-student scenario."""
    tic = time.time()
    # teacher generate training data
    X_train, y_train = teacher.generate_data(n_samples)
    # student fits model
    student.fit(X_train, y_train)
    # teacher generate test data
    X_test, y_test = teacher.generate_data(n_test)
    # train and test scores
    score_train = student.score(X_train, y_train)
    score_test = student.score(X_test, y_test)
    # duration
    tac = time.time()
    result = dict(
        score_train=score_train, score_test=score_test, elapsed_time=tac - tic
    )
    if return_all:
        add_result = dict(student=student, teacher=teacher, X=X_train, y=y_train)
        result.update(add_result)
    return result
