#!/usr/bin/env python3

import os
import sys
import logging
from datetime import datetime
import subprocess

# Configure logging to both file and console
def setup_logging():
    """Setup logging to both file and console."""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Clear the log file before starting
    log_file = 'logs/pipeline_execution.log'
    if os.path.exists(log_file):
        os.remove(log_file)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler
            logging.FileHandler(log_file, mode='w'),
            # Console handler
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def run_pipeline():
    """Run the pipeline with logging."""
    log_file = setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("STARTING PIPELINE EXECUTION")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("=" * 80)
    
    try:
        # Import and run the pipeline
        from orchestration.main import run_pipeline
        result = run_pipeline()
        
        logger.info("=" * 80)
        logger.info(f"PIPELINE COMPLETED WITH EXIT CODE: {result}")
        logger.info("=" * 80)
        
        return result
        
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}", exc_info=True)
        return 1

def show_log_tail(log_file, lines=200):
    """Show the last N lines of the log file."""
    try:
        result = subprocess.run(
            ['tail', f'-{lines}', log_file],
            capture_output=True,
            text=True,
            check=True
        )
        print("\n" + "=" * 80)
        print(f"LAST {lines} LINES OF LOG FILE:")
        print("=" * 80)
        print(result.stdout)
        print("=" * 80)
    except subprocess.CalledProcessError as e:
        print(f"Error reading log file: {e}")
    except FileNotFoundError:
        print("Log file not found")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Run the pipeline
    exit_code = run_pipeline()
    
    # Show log tail
    show_log_tail('logs/pipeline_execution.log', 200)
    
    # Exit with the same code
    sys.exit(exit_code)
