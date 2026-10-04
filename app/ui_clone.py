"""Presentation helpers for the CDH UI clone. No analytical or database logic."""
from contextlib import contextmanager
from pathlib import Path
import base64
import html
from urllib.parse import quote

import streamlit as st


def _stylesheet() -> str:
    root = Path(__file__).parent
    css = (root / "ui_clone.css").read_text(encoding="utf-8")
    fonts = []
    for weight in (400, 500, 600, 700):
        font = root / "assets" / f"inter-latin-{weight}-normal.woff2"
        if font.is_file():
            encoded = base64.b64encode(font.read_bytes()).decode("ascii")
            fonts.append(f"@font-face{{font-family:Inter;font-style:normal;font-weight:{weight};font-display:swap;src:url(data:font/woff2;base64,{encoded}) format('woff2');}}")
    icons = {
        "planning-overview": '<rect x="2" y="2" width="6" height="6" rx="1"/><rect x="12" y="2" width="6" height="6" rx="1"/><rect x="2" y="12" width="6" height="6" rx="1"/><rect x="12" y="12" width="6" height="6" rx="1"/>',
        "medicine-prioritization": '<path d="M7 2.5l9 5v9l-9 5-9-5v-9z" transform="translate(3 -2) scale(.85)"/>',
        "forecast-planning": '<path d="M2 17h16M3 14l5-5 3 3 6-8"/><path d="M14 4h3v3"/>',
        "procurement-planning": '<path d="M2 4h2l2.2 10h9.6l2-7H5"/><circle cx="8" cy="17" r="1"/><circle cx="15" cy="17" r="1"/>',
        "expiry-and-data-quality-review": '<rect x="2.5" y="4" width="15" height="14" rx="2"/><path d="M6 2v4m8-4v4M3 8h14M6 12h8M6 15h5"/>',
        "retrospective-validation": '<path d="M10 2l7 3v5c0 4-3 6-7 8-4-2-7-4-7-8V5z"/><path d="M7 10l2 2 4-4"/>',
        "methodology-and-governance": '<path d="M3 3h6c1.5 0 2.5.5 3 1.5C12.5 3.5 13.5 3 15 3h3v14h-3c-1.5 0-2.5.5-3 1.5C11.5 17.5 10.5 17 9 17H3zM12 4.5v14"/>',
    }
    nav_css = []
    for slug, paths in icons.items():
        svg = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' "
               "fill='none' stroke='black' stroke-width='1.5' stroke-linecap='round' "
               "stroke-linejoin='round'>" + paths + "</svg>")
        nav_css.append(
            f'.st-key-sidebar-nav-{slug} button::before'
            + '{--nav-icon:url("data:image/svg+xml,' + quote(svg, safe="") + '")}'
        )
    return "\n".join(fonts) + css + "\n" + "\n".join(nav_css)


def install_clone_style() -> None:
    st.html("<style>" + _stylesheet() + "</style>")


@contextmanager
def clone_section(title: str, key: str):
    """A consistently padded, keyboard-accessible collapsible section."""
    with st.container(key=f"clone-section-{key}"):
        with st.expander(title, expanded=True):
            yield


def clone_topbar(page: str) -> None:
    st.markdown(
        '<header id="planning-workspace" class="clone-topbar"><div class="clone-breadcrumb">'
        '<span class="clone-grid-icon" aria-hidden="true"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="2" width="6" height="6" rx="1"/><rect x="12" y="2" width="6" height="6" rx="1"/><rect x="2" y="12" width="6" height="6" rx="1"/><rect x="12" y="12" width="6" height="6" rx="1"/></svg></span>'
        '<span>Workspace</span><span aria-hidden="true">›</span>'
        f'<strong>{html.escape(page)}</strong></div>'
        '<div class="clone-topbar-tools"><span class="clone-data-status">'
        '<i></i>Preview data</span><span class="clone-topbar-sparkle" aria-hidden="true">✧</span><details class="clone-help"><summary>ⓘ &nbsp; Help</summary>'
        '<div>Use the sidebar to explore your planning data. Expand decision questions '
        'for supporting evidence. Forecasts are planning scenarios, not guaranteed outcomes.</div>'
        '</details><span class="clone-avatar" title="Pharmacy workspace">PH</span></div></header>',
        unsafe_allow_html=True,
    )


def clone_footer() -> None:
    st.markdown(
        '<footer class="clone-footer"><div><strong>Evidence-led planning. People-led decisions.</strong>'
        '<p>CDH Pharmaceutical Inventory Decision Support System</p></div>'
        '<a href="#planning-workspace" target="_self">Back to top &nbsp; ↑</a></footer>',
        unsafe_allow_html=True,
    )
