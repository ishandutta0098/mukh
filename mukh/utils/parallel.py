import multiprocessing as mp
from typing import Any, Callable, Iterable, Optional

from tqdm import tqdm


def get_cpu_count() -> int:
    """
    Get the number of CPU cores available.
    """
    return mp.cpu_count()


class MultiProcessor:
    """
    A multiprocessing utility with progress bar support.

    This class provides methods to execute functions on iterables either sequentially
    or in parallel with customizable progress tracking.
    """

    def __init__(
        self,
        num_processes: Optional[int] = 0,
        maintain_order: bool = False,
        initializer_func: Optional[Callable[..., Any]] = None,
        initializer_args: Iterable[Any] = (),
        start_mode: str = "fork",
        progress_bar_options: dict[str, Any] = None,
    ):
        """
        Initialize the ParallelProcessor.

        Args:
            num_processes: Number of parallel processes to use. Defaults to 0. Executes sequentially if set to 0.
            maintain_order: Whether to preserve the order of the input iterable in the results. Default is False.
            initializer_func: A function to execute at the start of each process.
            initializer_args: Arguments to pass to the initializer function.
            start_mode: Specifies the multiprocessing start method ("fork", "spawn", or "forkserver").
            progress_bar_options: Additional keyword arguments to customize the progress bar.
        """
        self.num_processes = num_processes
        self.maintain_order = maintain_order
        self.initializer_func = initializer_func
        self.initializer_args = initializer_args
        self.start_mode = start_mode
        self.progress_bar_options = progress_bar_options or {}

    def process(
        self,
        function: Callable[[Any], Any],
        iterable: Iterable[Any],
        description: str,
        total_elements: Optional[int] = None,
        gather_results: bool = True,
    ) -> Iterable[Any]:
        """
        Execute a function on each item of an iterable, optionally in parallel, with a progress bar.

        Args:
            function: The function to execute on each element. It should be a globally defined function.
            iterable: An iterable containing the elements to process.
            description: A label to display alongside the progress bar.
            total_elements: Total number of elements in the iterable. Useful if the iterable has no defined length.
            gather_results: If True, returns a list of results. Otherwise, yields results as they are processed.

        Returns:
            A list of results if `gather_results` is True; otherwise, an iterator yielding results one by one.

        Example:
            processor = ParallelProcessor(num_processes=4)
            results = processor.process(function, iterable, description)
        """

        def process_items():
            if self.num_processes != 0:
                pool = mp.get_context(self.start_mode).Pool(
                    self.num_processes, self.initializer_func, self.initializer_args
                )
            processor = (
                map
                if self.num_processes == 0
                else (pool.imap if self.maintain_order else pool.imap_unordered)
            )
            total = len(iterable) if hasattr(iterable, "__len__") else total_elements
            yield from tqdm(
                processor(function, iterable),
                desc=description,
                total=total,
                **self.progress_bar_options
            )
            if self.num_processes != 0:
                pool.terminate()

        if self.num_processes == 0 and self.initializer_func is not None:
            self.initializer_func(*self.initializer_args)

        return list(process_items()) if gather_results else process_items()
