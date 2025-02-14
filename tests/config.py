"""
Test configuration settings for Collatz conjecture verification.
"""

# Test parameters
TEST_PARAMS = {
    # Sample sizes for different test categories
    "small_sample": 100,
    "medium_sample": 1000,
    "large_sample": 10000,
    # Thresholds for statistical tests
    "p_value_threshold": 0.05,
    "error_margin": 0.1,
    "confidence_level": 0.95,
    # Performance test settings
    "timeout": 60,  # seconds
    "max_memory": 1024 * 1024 * 1024,  # 1GB in bytes
    # Verification limits
    "max_trajectory_length": 1000,
    "max_number": 1000000,
    # Theorem verification settings
    "theorem_test_cases": {
        "thm:one_way": {"limit": 1000},
        "thm:avalanche": {"limit": 1000},
        "thm:tau_dist": {"limit": 10000},
        "thm:measure_preserve": {"limit": 1000},
        "thm:ergodic": {"limit": 1000, "iterations": 100},
        "thm:entropy": {"limit": 1000},
        "thm:cycle_prevent": {"limit": 1000},
        "thm:global_descent": {"limit": 1000},
    },
}

# Test data paths
TEST_DATA = {
    "results_dir": "test_results",
    "cache_dir": "test_cache",
    "log_dir": "test_logs",
}

# Logging configuration
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
}

# Performance monitoring settings
PERFORMANCE_CONFIG = {
    "enable_profiling": True,
    "profile_output_dir": "profiles",
    "memory_tracking": True,
    "timing_stats": True,
}

# Test categories
TEST_CATEGORIES = {
    "unit": ["test_collatz.py"],
    "integration": ["test_theorem_mapper.py"],
    "performance": ["test_performance.py"],
    "statistical": ["test_measure_theory.py", "test_information_theory.py"],
}
