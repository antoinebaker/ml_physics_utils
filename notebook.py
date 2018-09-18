from ipywidgets import FloatProgress
from IPython.display import display
from contextlib import contextmanager


@contextmanager
def progressbar():
    progress = FloatProgress(min=0, max=1)
    display(progress)

    def on_progress(iteration, total):
        progress.value = iteration / total
    try:
        yield on_progress
    finally:
        print("Done")
