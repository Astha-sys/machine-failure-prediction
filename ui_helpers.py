from pathlib import Path
from contextlib import contextmanager
import streamlit as st

def apply_ui_theme():
    base = Path(__file__).parent / "assets" / "styles.css"
    if base.exists():
        css = base.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    else:
        st.warning("UI theme file not found: assets/styles.css")

@contextmanager
def glass_card(title: str = None):
    header_html = '<div class="glass-card">'
    if title:
        header_html = f'<div class="glass-card"><div class="card-header"><h3 class="card-title">{title}</h3></div>'
    st.markdown(header_html, unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)
