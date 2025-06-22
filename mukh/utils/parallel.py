from typing import Any, Callable, Iterable, Optional

import multiprocessing as mp
from tqdm import tqdm


def tqdm_parallel_processor(
    function: Callable[[Any], Any],
    iterable: Iterable[Any],
    description: str,
    num_processes: Optional[int] = None,
    maintain_order: bool = False,
    total_elements: Optional[int] = None,
    gather_results: bool = True,
    initializer_func: Optional[Callable[..., Any]] = None,
    initializer_args: Iterable[Any] = (),
    start_mode: str = "fork",
    progress_bar_options: dict[str, Any] = {},
) -> Iterable[Any]:
    """
    Execute a function on each item of an iterable, optionally in parallel, with a progress bar.

    Args:
        function: The function to execute on each element. It should be a globally defined function.
        iterable: An iterable containing the elements to process.
        description: A label to display alongside the progress bar.
        num_processes: Number of parallel processes to use. Defaults to the number of CPU cores if None.
            Executes sequentially if set to 0.
        maintain_order: Whether to preserve the order of the input iterable in the results. Default is False.
        total_elements: Total number of elements in the iterable. Useful if the iterable has no defined length.
        gather_results: If True, returns a list of results. Otherwise, yields results as they are processed.
        initializer_func: A function to execute at the start of each process.
        initializer_args: Arguments to pass to the initializer function.
        start_mode: Specifies the multiprocessing start method ("fork", "spawn", or "forkserver").
        progress_bar_options: Additional keyword arguments to customize the progress bar.

    Returns:
        A list of results if `gather_results` is True; otherwise, an iterator yielding results one by one.
    """

    def process_items():
        if num_processes != 0:
            pool = mp.get_context(start_mode).Pool(num_processes, initializer_func, initializer_args)
        processor = map if num_processes == 0 else (pool.imap if maintain_order else pool.imap_unordered)
        total = len(iterable) if hasattr(iterable, "__len__") else total_elements
        yield from tqdm(processor(function, iterable), desc=description, total=total, **progress_bar_options)
        if num_processes != 0:
            pool.terminate()

    if num_processes == 0 and initializer_func is not None:
        initializer_func(*initializer_args)

    return list(process_items()) if gather_results else process_items()
