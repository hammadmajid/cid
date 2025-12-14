import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile


def parse_uploaded_file(uploaded_file: UploadedFile | None) -> str:
    """Parse uploaded file and return text content."""
    if uploaded_file is None:
        return ""

    if uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")
        st.sidebar.success(f"Loaded {len(text)} characters")
        return text

    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.sidebar.success(f"Loaded {len(df)} rows")
        text_columns = df.select_dtypes(include=["object"]).columns
        if len(text_columns) > 0:
            return " ".join(df[text_columns[0]].astype(str).tolist())

    return ""
