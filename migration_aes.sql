-- Run ONCE in MySQL Workbench (database: legal_analysis_db).
-- Encrypted text is ~1.4x longer than the original, so the column must be LONGTEXT.
ALTER TABLE cases MODIFY extracted_text LONGTEXT;
