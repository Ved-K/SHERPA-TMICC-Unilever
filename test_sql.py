import os
import pyodbc

conn_str = os.getenv("AZURE_SQL_CONN_STR")
print("conn_str type:", type(conn_str))
print("conn_str starts:", conn_str[:60])

con = pyodbc.connect(conn_str)
cur = con.cursor()
cur.execute("SELECT 1")
print("OK:", cur.fetchone())
con.close()
