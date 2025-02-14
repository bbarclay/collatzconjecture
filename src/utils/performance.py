"""
Performance monitoring and optimization tools for Collatz verification.
"""

import time
import cProfile
import pstats
import io
import logging
import functools
import psutil
import tracemalloc
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from datetime import datetime

from tests.config import PERFORMANCE_CONFIG, TEST_DATA


class PerformanceMonitor:
    """Monitors and logs performance metrics for critical functions"""

    def __init__(self):
        self.stats = {"function_calls": {}, "memory_usage": {}, "execution_times": {}}
        self._setup_logging()

    def _setup_logging(self):
        """Set up performance logging"""
        log_dir = Path(TEST_DATA["log_dir"])
        log_dir.mkdir(exist_ok=True)

        self.logger = logging.getLogger("PerformanceMonitor")
        if not self.logger.handlers:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_dir / f"performance_{timestamp}.log"

            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def profile(self, func: Callable) -> Callable:
        """Decorator for profiling function execution"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Start profiling
            profiler = cProfile.Profile()
            start_time = time.time()
            tracemalloc.start()

            # Execute function
            try:
                result = profiler.runcall(func, *args, **kwargs)
            finally:
                # Collect metrics
                end_time = time.time()
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                # Process profiling data
                s = io.StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                ps.print_stats()

                # Record statistics
                func_name = func.__name__
                self.stats["function_calls"].setdefault(func_name, []).append(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "execution_time": end_time - start_time,
                        "current_memory": current / 1024 / 1024,  # MB
                        "peak_memory": peak / 1024 / 1024,  # MB
                        "profile": s.getvalue(),
                    }
                )

                # Log performance data
                self.logger.info(
                    f"Function: {func_name}, "
                    f"Time: {end_time - start_time:.3f}s, "
                    f"Memory: {peak/1024/1024:.1f}MB"
                )

            return result

        return wrapper

    def track_memory(self, func: Callable) -> Callable:
        """Decorator for tracking memory usage"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process()
            start_memory = process.memory_info().rss

            result = func(*args, **kwargs)

            end_memory = process.memory_info().rss
            memory_used = end_memory - start_memory

            func_name = func.__name__
            self.stats["memory_usage"].setdefault(func_name, []).append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "memory_used": memory_used / 1024 / 1024,  # MB
                }
            )

            return result

        return wrapper

    def time_execution(self, func: Callable) -> Callable:
        """Decorator for timing function execution"""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            func_name = func.__name__
            self.stats["execution_times"].setdefault(func_name, []).append(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "execution_time": execution_time,
                }
            )

            return result

        return wrapper

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report from collected statistics"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {},
            "details": self.stats,
        }

        # Compute summary statistics
        for func_name, calls in self.stats["function_calls"].items():
            times = [call["execution_time"] for call in calls]
            memory = [call["peak_memory"] for call in calls]

            report["summary"][func_name] = {
                "calls": len(calls),
                "avg_time": sum(times) / len(times),
                "max_time": max(times),
                "avg_memory": sum(memory) / len(memory),
                "max_memory": max(memory),
            }

        return report

    def save_report(self, report: Optional[Dict[str, Any]] = None) -> Path:
        """Save performance report to file"""
        if report is None:
            report = self.get_performance_report()

        profile_dir = Path(PERFORMANCE_CONFIG["profile_output_dir"])
        profile_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = profile_dir / f"performance_report_{timestamp}.json"

        with open(output_file, "w") as f:
            import json

            json.dump(report, f, indent=2)

        return output_file


# Global performance monitor instance
monitor = PerformanceMonitor()


# Convenience decorators
def profile(func):
    """Profile function execution if profiling is enabled"""
    if PERFORMANCE_CONFIG["enable_profiling"]:
        return monitor.profile(func)
    return func


def track_memory(func):
    """Track function memory usage if memory tracking is enabled"""
    if PERFORMANCE_CONFIG["memory_tracking"]:
        return monitor.track_memory(func)
    return func


def time_execution(func):
    """Time function execution if timing stats are enabled"""
    if PERFORMANCE_CONFIG["timing_stats"]:
        return monitor.time_execution(func)
    return func
