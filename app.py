import streamlit as st

from src.config import PAGE_CONFIG, STYLES
from src.utils import parse_uploaded_file
from src.pages import render_dashboard, render_analysis, render_settings


def main():
    """Main application entry point."""
    st.set_page_config(**PAGE_CONFIG)
    st.markdown(STYLES, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select page:",
        ["Dashboard", "Analysis", "Settings"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload data (CSV/TXT)",
        type=["csv", "txt"],
        help="Upload text data for analysis",
    )

    uploaded_text = parse_uploaded_file(uploaded_file)

    st.sidebar.markdown("---")
    st.sidebar.caption("Competitive Intelligence Dashboard")

    # Route to pages
    if page == "Dashboard":
        render_dashboard()
    elif page == "Analysis":
        render_analysis(uploaded_text)
    elif page == "Settings":
        render_settings()

    # Footer
    st.markdown("---")
    st.caption("Competitive Intelligence Dashboard")


if __name__ == "__main__":
    main()
