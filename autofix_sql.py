import store
import os
import streamlit as st

conn_str = os.getenv("AZURE_SQL_CONN_STR") or st.secrets["AZURE_SQL_CONN_STR"]
con = store.connect(conn_str, init_schema=True)  # rebuilds / adds missing columns
print("Schema refreshed successfully ✅")
