-- Dynamic Stage 0 - Database Initialization Script
-- This script creates the database schema for the GPU-accelerated feature stationarization pipeline

-- Create database if it doesn't exist
CREATE DATABASE IF NOT EXISTS dynamic_stage0_db
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE dynamic_stage0_db;

-- Table 1: processing_tasks
-- Serves as the source of truth for work to be done
CREATE TABLE IF NOT EXISTS processing_tasks (
    task_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for the task',
    currency_pair VARCHAR(16) NOT NULL UNIQUE COMMENT 'Currency pair to be processed (e.g., EURUSD)',
    r2_path VARCHAR(1024) NOT NULL COMMENT 'Base path in R2 for this pair data',
    added_timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'When the task was added',

    -- Indexes for performance
    INDEX idx_currency_pair (currency_pair),
    INDEX idx_added_timestamp (added_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Defines the list of currency pairs to be processed by the pipeline';

-- Table 2: feature_status
-- Acts as a detailed audit log capturing execution history
CREATE TABLE IF NOT EXISTS feature_status (
    status_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for the status entry',
    task_id INT NOT NULL COMMENT 'Links to the specific task',
    status ENUM('PENDING', 'RUNNING', 'COMPLETED', 'FAILED') NOT NULL COMMENT 'Current processing status',
    start_time DATETIME NULL COMMENT 'Timestamp when processing for this pair started',
    end_time DATETIME NULL COMMENT 'Timestamp when processing for this pair ended',
    hostname VARCHAR(255) NULL COMMENT 'ID of the vast.ai instance or hostname running the task',
    error_message TEXT NULL COMMENT 'Detailed error message and traceback if status is FAILED',

    -- Adicionada constraint UNIQUE para garantir que cada tarefa tenha apenas um status
    UNIQUE KEY uq_task_id (task_id),

    -- Foreign key constraint
    FOREIGN KEY (task_id) REFERENCES processing_tasks(task_id) ON DELETE CASCADE,

    -- Indexes for performance
    INDEX idx_status (status),
    INDEX idx_start_time (start_time),
    INDEX idx_hostname (hostname)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
COMMENT='Tracks the execution status and history for each processing task';

-- Insert sample data for testing (optional)
-- Uncomment the following lines to add sample currency pairs for testing

/*
INSERT INTO processing_tasks (currency_pair, r2_path) VALUES
('EURUSD', 'data/forex/EURUSD/'),
('GBPUSD', 'data/forex/GBPUSD/'),
('USDJPY', 'data/forex/USDJPY/')
ON DUPLICATE KEY UPDATE r2_path = VALUES(r2_path);
*/

-- Create a view for easy querying of pending tasks
CREATE OR REPLACE VIEW pending_tasks AS
SELECT
    pt.task_id,
    pt.currency_pair,
    pt.r2_path,
    pt.added_timestamp,
    COALESCE(fs.status, 'PENDING') as current_status,
    fs.start_time,
    fs.end_time,
    fs.hostname,
    fs.error_message
FROM processing_tasks pt
LEFT JOIN feature_status fs ON pt.task_id = fs.task_id
WHERE fs.status IS NULL
   OR fs.status IN ('PENDING', 'FAILED')
   OR (fs.status = 'RUNNING' AND fs.start_time < DATE_SUB(NOW(), INTERVAL 1 HOUR));

-- Create a view for completed tasks
CREATE OR REPLACE VIEW completed_tasks AS
SELECT
    pt.task_id,
    pt.currency_pair,
    pt.r2_path,
    pt.added_timestamp,
    fs.status,
    fs.start_time,
    fs.end_time,
    fs.hostname,
    TIMESTAMPDIFF(SECOND, fs.start_time, fs.end_time) as processing_time_seconds
FROM processing_tasks pt
INNER JOIN feature_status fs ON pt.task_id = fs.task_id
WHERE fs.status = 'COMPLETED';

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON dynamic_stage0_db.* TO 'dynamic_stage0_user'@'%';
-- FLUSH PRIVILEGES;
