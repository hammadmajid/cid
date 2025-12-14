import streamlit as st


def header(title: str, subtitle: str = ""):
    """Render page header."""
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sub-header">{subtitle}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, help_text: str = ""):
    """Render a metric card."""
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(label=label, value=value, help=help_text)
    st.markdown("</div>", unsafe_allow_html=True)


def entity_badges(entities: list[str], empty_message: str = "None detected"):
    """Render entity badges."""
    if entities:
        for entity in entities:
            st.markdown(
                f'<span class="entity-badge">{entity}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.info(empty_message)
