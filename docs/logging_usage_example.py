#!/usr/bin/env python3
"""
Example demonstrating the new structured logging system.

This example shows how to use the contextual logging utilities
to create structured, event-based logs without "Stage" numbering.
"""

import logging
import logging.config
import yaml
import time
from pathlib import Path

# Import our new logging utilities
from utils.log_context import set_run_id, set_task, set_engine, set_cluster_info, bind_context, with_context
from utils.logging_utils import (
    get_logger, info_event, warn_event, error_event, critical_event, 
    time_event, Events
)


def setup_logging():
    """Set up logging configuration."""
    config_path = Path(__file__).parent.parent / "config" / "logging.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.config.dictConfig(config)


def example_pipeline_execution():
    """Example of how the new logging system would work in practice."""
    
    # Set up logging
    setup_logging()
    
    # Get contextual loggers
    main_logger = get_logger(__name__, "orchestration.main")
    orchestrator_logger = get_logger("orchestration.orchestrator")
    processor_logger = get_logger("orchestration.processor")
    engine_logger = get_logger("features.FeatureEngineeringEngine")
    
    # Set run context
    set_run_id(42)
    set_cluster_info(hostname="worker-01", gpu_count=4, workers=8, 
                    dashboard_url="http://localhost:8787")
    
    # Pipeline start
    info_event(main_logger, Events.PIPELINE_START, 
               "Starting pipeline execution", 
               hostname="worker-01")
    
    # Task discovery
    info_event(orchestrator_logger, Events.TASK_DISCOVERY_START, 
               "Starting task discovery")
    
    # Simulate finding tasks
    time.sleep(0.1)
    info_event(orchestrator_logger, Events.TASK_DISCOVERY_FOUND, 
               "Found 8 tasks to process", count=8, path="/data/forex")
    
    # Cluster ready
    info_event(orchestrator_logger, Events.CLUSTER_READY, 
               "Cluster is ready for processing", 
               gpu_count=4, workers=8, dashboard_url="http://localhost:8787")
    
    # Process tasks
    info_event(orchestrator_logger, Events.TASK_EXECUTION_START, 
               "Starting task execution phase")
    
    # Example task processing
    for i, pair in enumerate(["EURUSD", "GBPUSD", "USDJPY"], 1):
        set_task(task_id=100 + i, pair=pair)
        
        info_event(processor_logger, Events.TASK_START, 
                   f"Processing pair {pair}", 
                   index=i, total=3, pair=pair, 
                   file=f"{pair}_2023.feather", size_mb=512.4)
        
        # Engine processing with timing
        with time_event(engine_logger, Events.ENGINE_START, 
                       f"Processing {pair} with BK filter", 
                       pair=pair, order=2, desc="BK filter"):
            
            # Simulate engine work
            time.sleep(0.2)
            
            # Log engine completion
            info_event(engine_logger, Events.ENGINE_END, 
                       f"Engine processing completed for {pair}", 
                       pair=pair, order=2, desc="BK filter",
                       cols_before=120, cols_after=158, new_cols=38)
        
        # I/O operations
        info_event(processor_logger, Events.IO_SAVE_START, 
                   f"Saving processed data for {pair}", 
                   pair=pair, parts=32, path=f"/out/{pair}/")
        
        time.sleep(0.1)  # Simulate save operation
        
        info_event(processor_logger, Events.IO_SAVE_END, 
                   f"Data saved successfully for {pair}", 
                   pair=pair, parts=32, path=f"/out/{pair}/")
        
        # Task completion
        info_event(processor_logger, Events.TASK_SUCCESS, 
                   f"Task completed successfully for {pair}", 
                   pair=pair)
    
    # Pipeline summary
    info_event(main_logger, Events.PIPELINE_SUMMARY, 
               "Pipeline execution completed", 
               total=3, success=3, failed=0)


def example_with_decorator():
    """Example using the bind_context decorator."""
    
    setup_logging()
    logger = get_logger(__name__)
    
    @bind_context(run_id=123, pair="EURUSD", engine="FeatureEngineering")
    def process_currency_pair():
        """Process a currency pair with bound context."""
        info_event(logger, Events.ENGINE_START, 
                   "Processing currency pair with bound context")
        
        # The context is automatically available in all logging calls
        time.sleep(0.1)
        
        info_event(logger, Events.ENGINE_END, 
                   "Currency pair processing completed")
    
    process_currency_pair()


def example_with_context_manager():
    """Example using the with_context context manager."""
    
    setup_logging()
    logger = get_logger(__name__)
    
    with with_context(run_id=456, pair="GBPUSD", engine="StatisticalTests"):
        info_event(logger, Events.ENGINE_START, 
                   "Processing with context manager")
        
        time.sleep(0.1)
        
        info_event(logger, Events.ENGINE_END, 
                   "Context manager processing completed")


def example_error_handling():
    """Example of error logging with the new system."""
    
    setup_logging()
    logger = get_logger(__name__)
    
    set_run_id(789)
    set_task(task_id=200, pair="USDJPY")
    
    try:
        # Simulate an error condition
        raise ValueError("Simulated processing error")
    except Exception as e:
        error_event(logger, Events.TASK_FAILURE, 
                    f"Task failed with error: {str(e)}", 
                    error_type=type(e).__name__, 
                    error_message=str(e))


if __name__ == "__main__":
    print("=== New Structured Logging System Examples ===\n")
    
    print("1. Pipeline Execution Example:")
    print("-" * 40)
    example_pipeline_execution()
    
    print("\n2. Context Decorator Example:")
    print("-" * 40)
    example_with_decorator()
    
    print("\n3. Context Manager Example:")
    print("-" * 40)
    example_with_context_manager()
    
    print("\n4. Error Handling Example:")
    print("-" * 40)
    example_error_handling()
    
    print("\n=== Examples completed ===")
    print("Check logs/pipeline_execution.log for structured JSON output")
    print("Check logs/pipeline_errors.log for error details")
