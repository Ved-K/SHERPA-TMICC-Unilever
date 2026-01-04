CREATE TABLE app_users (
  id            INT IDENTITY(1,1) PRIMARY KEY,
  email         NVARCHAR(255) NOT NULL UNIQUE,
  role          NVARCHAR(32)  NOT NULL DEFAULT 'user',   -- 'admin' | 'user'
  is_active     BIT           NOT NULL DEFAULT 1,
  created_at    DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME(),
  created_by    NVARCHAR(255) NULL,
  updated_at    DATETIME2     NOT NULL DEFAULT SYSUTCDATETIME()
);
GO
