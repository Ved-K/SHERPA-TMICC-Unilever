-- Add missing columns safely (idempotent)

IF COL_LENGTH('machines', 'sort_index') IS NULL
    ALTER TABLE machines ADD sort_index INT NOT NULL CONSTRAINT DF_machines_sort_index DEFAULT 0;

IF COL_LENGTH('tasks', 'sort_index') IS NULL
    ALTER TABLE tasks ADD sort_index INT NOT NULL CONSTRAINT DF_tasks_sort_index DEFAULT 0;

IF COL_LENGTH('steps', 'sort_index') IS NULL
    ALTER TABLE steps ADD sort_index INT NOT NULL CONSTRAINT DF_steps_sort_index DEFAULT 0;

IF COL_LENGTH('machines', 'machine_type') IS NULL
    ALTER TABLE machines ADD machine_type NVARCHAR(255) NOT NULL CONSTRAINT DF_machines_machine_type DEFAULT '';

IF COL_LENGTH('tasks', 'phases_json') IS NULL
    ALTER TABLE tasks ADD phases_json NVARCHAR(MAX) NOT NULL CONSTRAINT DF_tasks_phases_json DEFAULT '[]';

IF COL_LENGTH('steps', 'eng_controls') IS NULL
    ALTER TABLE steps ADD eng_controls NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_eng_controls DEFAULT '';

IF COL_LENGTH('steps', 'admin_controls') IS NULL
    ALTER TABLE steps ADD admin_controls NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_admin_controls DEFAULT '';

IF COL_LENGTH('steps', 'hazard_text') IS NULL
    ALTER TABLE steps ADD hazard_text NVARCHAR(MAX) NOT NULL CONSTRAINT DF_steps_hazard_text DEFAULT '';
