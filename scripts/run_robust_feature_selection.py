#!/usr/bin/env python3
"""
Robust Feature Selection Pipeline Executor

This script executes the complete robust feature selection pipeline:
1. Runs Optuna Study A with walk-forward validation and trading metrics
2. Analyzes feature robustness from the results
3. Generates comprehensive reports and recommendations

Usage:
    python run_robust_feature_selection.py --config config/study/stageA.yaml --output-dir ./robust_results
"""

import argparse
import logging
import sys
from pathlib import Path
import subprocess
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('robust_feature_selection.log')
        ]
    )

def run_optuna_study(config_path: str, output_dir: Path) -> Path:
    """
    Run the Optuna Study A with robust feature selection.
    
    Args:
        config_path: Path to the Hydra configuration file
        output_dir: Output directory for results
        
    Returns:
        Path to the SQLite study database
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Optuna Study A with robust feature selection...")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the main orchestration script
    try:
        # Assuming the main script is orchestration/main.py
        main_script = project_root / "orchestration" / "main.py"
        
        cmd = [
            sys.executable, str(main_script),
            "--config-path", str(Path(config_path).parent),
            "--config-name", Path(config_path).stem,
            f"output.output_path={output_dir}",
            "study.mode=preprocess_selection"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        if result.returncode != 0:
            logger.error(f"Study execution failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Study execution failed: {result.stderr}")
        
        logger.info("Study execution completed successfully")
        logger.info(f"STDOUT: {result.stdout}")
        
        # Find the SQLite database
        study_db_path = output_dir / "optuna_stage" / "study_a.sqlite"
        
        if not study_db_path.exists():
            # Try alternative locations
            possible_paths = [
                output_dir / "study_a.sqlite",
                output_dir / "optuna_stage" / "stageA_preprocess_selection.sqlite"
            ]
            
            for path in possible_paths:
                if path.exists():
                    study_db_path = path
                    break
            else:
                raise FileNotFoundError(f"Could not find study database. Checked: {[study_db_path] + possible_paths}")
        
        logger.info(f"Study database found at: {study_db_path}")
        return study_db_path
        
    except Exception as e:
        logger.error(f"Failed to run Optuna study: {e}")
        raise

def run_feature_analysis(study_db_path: Path, output_dir: Path, 
                        frequency_threshold: float = 0.5,
                        stability_threshold: float = 0.7) -> None:
    """
    Run feature robustness analysis on the study results.
    
    Args:
        study_db_path: Path to the SQLite study database
        output_dir: Output directory for analysis results
        frequency_threshold: Minimum frequency for feature recommendation
        stability_threshold: Minimum stability for feature recommendation
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Running feature robustness analysis...")
    
    analysis_dir = output_dir / "feature_analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Run the feature analysis script
        analysis_script = project_root / "scripts" / "feature_robustness_analysis.py"
        
        cmd = [
            sys.executable, str(analysis_script),
            "--study-path", str(study_db_path),
            "--output-dir", str(analysis_dir),
            "--frequency-threshold", str(frequency_threshold),
            "--stability-threshold", str(stability_threshold)
        ]
        
        logger.info(f"Running analysis command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root)
        
        if result.returncode != 0:
            logger.error(f"Analysis failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Analysis failed: {result.stderr}")
        
        logger.info("Feature analysis completed successfully")
        logger.info(f"STDOUT: {result.stdout}")
        
    except Exception as e:
        logger.error(f"Failed to run feature analysis: {e}")
        raise

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Run robust feature selection pipeline')
    parser.add_argument('--config', default='config/study/stageA.yaml', 
                       help='Path to Hydra configuration file')
    parser.add_argument('--output-dir', default='./robust_results', 
                       help='Output directory for all results')
    parser.add_argument('--frequency-threshold', type=float, default=0.5,
                       help='Minimum frequency for feature recommendation')
    parser.add_argument('--stability-threshold', type=float, default=0.7,
                       help='Minimum stability for feature recommendation')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--skip-study', action='store_true',
                       help='Skip study execution and only run analysis (requires existing study database)')
    
    args = parser.parse_args()
    
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*60)
    logger.info("ROBUST FEATURE SELECTION PIPELINE")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Frequency threshold: {args.frequency_threshold}")
    logger.info(f"Stability threshold: {args.stability_threshold}")
    logger.info("="*60)
    
    start_time = time.time()
    
    try:
        if not args.skip_study:
            # Step 1: Run Optuna Study
            logger.info("STEP 1: Running Optuna Study A with robust validation...")
            study_db_path = run_optuna_study(args.config, output_dir)
        else:
            # Look for existing study database
            possible_paths = [
                output_dir / "optuna_stage" / "study_a.sqlite",
                output_dir / "study_a.sqlite",
                output_dir / "optuna_stage" / "stageA_preprocess_selection.sqlite"
            ]
            
            study_db_path = None
            for path in possible_paths:
                if path.exists():
                    study_db_path = path
                    break
            
            if not study_db_path:
                raise FileNotFoundError(f"No existing study database found. Checked: {possible_paths}")
            
            logger.info(f"Using existing study database: {study_db_path}")
        
        # Step 2: Analyze Feature Robustness
        logger.info("STEP 2: Analyzing feature robustness...")
        run_feature_analysis(
            study_db_path, 
            output_dir, 
            args.frequency_threshold, 
            args.stability_threshold
        )
        
        elapsed_time = time.time() - start_time
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Total execution time: {elapsed_time:.2f} seconds")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("="*60)
        
        # Print summary of results
        analysis_dir = output_dir / "feature_analysis"
        if (analysis_dir / "feature_analysis_report.md").exists():
            logger.info(f"📊 Comprehensive report: {analysis_dir / 'feature_analysis_report.md'}")
        if (analysis_dir / "feature_recommendations.csv").exists():
            logger.info(f"🎯 Feature recommendations: {analysis_dir / 'feature_recommendations.csv'}")
        if (analysis_dir / "analysis_results.json").exists():
            logger.info(f"📋 Detailed results: {analysis_dir / 'analysis_results.json'}")
        
        logger.info("\n🚀 Next steps:")
        logger.info("   1. Review the feature analysis report")
        logger.info("   2. Implement recommended features in your models")
        logger.info("   3. Monitor feature stability over time")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
