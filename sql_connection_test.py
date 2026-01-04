import pyodbc

conn_str = "DRIVER={ODBC Driver 18 for SQL Server};SERVER=tcp:sherpatestdb.database.windows.net,1433;DATABASE=SherpaDB;UID=sqladmin;PWD=Browassupcunt6969;Encrypt=yes;TrustServerCertificate=no;Connection Timeout=60;"
con = pyodbc.connect(conn_str)
print("Connected OK!")
