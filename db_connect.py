import pyodbc
import streamlit as st


def get_connection():
    conn_str = st.secrets["azure_sql"]["connection_string"]
    return pyodbc.connect(conn_str)
