-- schema.sql
-- Run this once in MySQL to create the database and tables manually.
-- (models.py also creates these automatically via SQLAlchemy on app startup,
--  this file is kept for reference / manual DB administration.)

CREATE DATABASE IF NOT EXISTS legal_analysis_db;
USE legal_analysis_db;

CREATE TABLE IF NOT EXISTS legal_provisions (
    id INT PRIMARY KEY AUTO_INCREMENT,
    act_name VARCHAR(255),
    section_number VARCHAR(50),
    title VARCHAR(500),
    full_text TEXT,
    category VARCHAR(100),
    source_url VARCHAR(500),
    embedding_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cases (
    id INT PRIMARY KEY AUTO_INCREMENT,
    file_path VARCHAR(500),
    extracted_text LONGTEXT,
    extraction_method ENUM('direct', 'ocr'),
    predicted_category VARCHAR(100),
    category_confidence FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS case_provision_matches (
    id INT PRIMARY KEY AUTO_INCREMENT,
    case_id INT,
    provision_id INT,
    relevance_score FLOAT,
    explanation TEXT,
    FOREIGN KEY (case_id) REFERENCES cases(id),
    FOREIGN KEY (provision_id) REFERENCES legal_provisions(id)
);
