-- 1) create table if it doesn't exist (safe)
IF OBJECT_ID('dbo.app_users', 'U') IS NULL
BEGIN
  CREATE TABLE dbo.app_users (
    email NVARCHAR(255) NOT NULL PRIMARY KEY,
    role NVARCHAR(50) NOT NULL,
    is_active BIT NOT NULL CONSTRAINT DF_app_users_is_active DEFAULT(1),
    created_by NVARCHAR(255) NULL,
    created_at DATETIME2 NOT NULL CONSTRAINT DF_app_users_created_at DEFAULT SYSUTCDATETIME(),
    updated_at DATETIME2 NULL
  );
END;

-- 2) add/activate yourself as admin
MERGE dbo.app_users AS t
USING (SELECT LOWER('vedkhanolkar7@gmail.com') AS email) s
ON t.email = s.email
WHEN MATCHED THEN
  UPDATE SET role='admin', is_active=1, updated_at=SYSUTCDATETIME()
WHEN NOT MATCHED THEN
  INSERT (email, role, is_active, created_by)
  VALUES (s.email, 'admin', 1, 'bootstrap');
