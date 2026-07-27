"""
PitWall — Streamlit Frontend
Premium F1-themed chat interface for the 2026 Regulations RAG consultant.
Run from the project root: streamlit run frontend/app.py
"""

import streamlit as st
import requests
import uuid
import os
import json
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PitWall | F1 2026 Regulations",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
API_BASE       = os.getenv("FASTAPI_URL", "http://localhost:8000")
CHAT_ENDPOINT  = f"{API_BASE}/chat"
HEALTH_ENDPOINT = f"{API_BASE}/health"
CLEAR_ENDPOINT = f"{API_BASE}/session"

# ---------------------------------------------------------------------------
# Premium F1 CSS — dark carbon/red theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Rajdhani:wght@500;600;700&display=swap');

/* ── Root palette ── */
:root {
    --f1-red:       #E10600;
    --f1-red-dark:  #A10400;
    --f1-red-glow:  rgba(225, 6, 0, 0.18);
    --carbon:       #0D0D0D;
    --carbon-2:     #141414;
    --carbon-3:     #1C1C1C;
    --carbon-4:     #242424;
    --carbon-5:     #2E2E2E;
    --text-primary: #F0F0F0;
    --text-muted:   #888888;
    --text-dim:     #555555;
    --border:       #2A2A2A;
    --border-accent:#3A3A3A;
}

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: var(--carbon) !important;
    color: var(--text-primary) !important;
}

/* ── Hide Streamlit chrome (keep header for sidebar toggle) ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { visibility: hidden; }
.stDeployButton { display: none; }
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 3rem !important;
}

/* ── Sidebar expand/collapse toggle button ── */
[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    background: var(--carbon-3) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 0 8px 8px 0 !important;
    color: var(--f1-red) !important;
    transition: background 0.2s ease, box-shadow 0.2s ease !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
}
[data-testid="collapsedControl"]:hover {
    background: var(--f1-red) !important;
    box-shadow: 2px 0 12px var(--f1-red-glow) !important;
    color: #fff !important;
}
[data-testid="collapsedControl"] svg {
    stroke: currentColor !important;
    fill: none !important;
}

/* ── Main content padding ── */
.block-container {
    padding: 1.5rem 2rem 4rem 2rem !important;
    max-width: 900px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111111 0%, #0D0D0D 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.5rem;
}

/* ── Sidebar logo block ── */
.pitwall-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.pitwall-logo-icon { font-size: 2rem; line-height: 1; }
.pitwall-logo-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 1px;
}
.pitwall-logo-text span { color: var(--f1-red); }

/* ── Header bar ── */
.chat-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 1rem 0 1.25rem 0;
    border-bottom: 2px solid var(--f1-red);
    margin-bottom: 1.5rem;
}
.chat-header-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.5px;
    margin: 0;
}
.chat-header-sub {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin: 0;
    margin-top: 2px;
}
.red-stripe {
    width: 4px;
    height: 44px;
    background: linear-gradient(180deg, var(--f1-red) 0%, var(--f1-red-dark) 100%);
    border-radius: 2px;
    flex-shrink: 0;
}

/* ── Status badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}
.status-online  { background: rgba(0,210,90,0.12); color: #00D25A; border: 1px solid rgba(0,210,90,0.25); }
.status-offline { background: rgba(225,6,0,0.12);  color: var(--f1-red); border: 1px solid rgba(225,6,0,0.25); }
.status-dot { width: 6px; height: 6px; border-radius: 50%; }
.status-dot-online  { background: #00D25A; }
.status-dot-offline { background: var(--f1-red); }

/* ── Sidebar section title ── */
.sidebar-section-title {
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin: 1.2rem 0 0.6rem 0;
}

/* ── Regulation pill ── */
.reg-pill {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 10px;
    margin-bottom: 5px;
    border-radius: 6px;
    background: var(--carbon-3);
    border: 1px solid var(--border);
    font-size: 0.78rem;
    color: var(--text-muted);
}
.reg-pill-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--f1-red); flex-shrink: 0; }

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: var(--carbon-3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.75rem !important;
    transition: border-color 0.2s ease;
}
[data-testid="stChatMessage"]:hover { border-color: var(--border-accent) !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--f1-red) !important;
    background: var(--carbon-4) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: var(--carbon-3) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--f1-red) !important;
    box-shadow: 0 0 0 3px var(--f1-red-glow) !important;
}
[data-testid="stChatInput"] textarea { color: var(--text-primary) !important; background: transparent !important; }
[data-testid="stChatInputSubmitButton"] { color: var(--f1-red) !important; }

/* ── Buttons ── */
.stButton > button {
    background: var(--carbon-4) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-accent) !important;
    border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s ease !important;
    width: 100%;
}
.stButton > button:hover {
    background: var(--f1-red) !important;
    border-color: var(--f1-red) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px var(--f1-red-glow) !important;
    transform: translateY(-1px);
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 1rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--carbon); }
::-webkit-scrollbar-thumb { background: var(--carbon-5); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--f1-red); }

/* ── Empty state ── */
.empty-state { text-align: center; padding: 4rem 2rem; color: var(--text-muted); }
.empty-state-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-state-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.empty-state-sub { font-size: 0.85rem; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "session_id"     not in st.session_state: st.session_state.session_id     = str(uuid.uuid4())
if "messages"       not in st.session_state: st.session_state.messages       = []
if "pending_prompt" not in st.session_state: st.session_state.pending_prompt = None


# ---------------------------------------------------------------------------
# Helper: backend health check (cached 30 s)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def check_health():
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="pitwall-logo">
        <div class="pitwall-logo-icon">🏎️</div>
        <div class="pitwall-logo-text">PIT<span>WALL</span></div>
    </div>
    """, unsafe_allow_html=True)

    is_online = check_health()
    if is_online:
        st.markdown('<div class="status-badge status-online"><div class="status-dot status-dot-online"></div> Backend Online</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-badge status-offline"><div class="status-dot status-dot-offline"></div> Backend Offline</div>', unsafe_allow_html=True)

    st.markdown('<p class="sidebar-section-title">📂 Loaded Regulations</p>', unsafe_allow_html=True)
    for code, name in [("A","General Provisions"),("B","Sporting"),("C","Technical"),
                       ("D","Financial — F1 Teams"),("E","Financial — PU Manufacturers"),("F","Operational")]:
        st.markdown(f'<div class="reg-pill"><div class="reg-pill-dot"></div><span><strong>Section {code}</strong> — {name}</span></div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<p class="sidebar-section-title">⚙️ Session</p>', unsafe_allow_html=True)
    st.markdown(f"<p style='font-size:0.72rem;color:#555;margin-bottom:0.6rem;word-break:break-all;'>{st.session_state.session_id[:18]}…</p>", unsafe_allow_html=True)

    if st.button("🔄 New Session", key="new_session"):
        try:
            requests.delete(f"{CLEAR_ENDPOINT}/{st.session_state.session_id}", timeout=3)
        except Exception:
            pass
        st.session_state.session_id     = str(uuid.uuid4())
        st.session_state.messages       = []
        st.session_state.pending_prompt = None
        check_health.clear()
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <p style='font-size:0.7rem;color:#3A3A3A;line-height:1.6;'>
    FIA 2026 F1 Regulations<br>All Sections · Issue June 2026<br>
    Powered by Nemotron Ultra · OpenRouter<br><br>
    <em>Not an official FIA product.</em>
    </p>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="chat-header">
    <div class="red-stripe"></div>
    <div>
        <p class="chat-header-title">F1 2026 Regulations Consultant</p>
        <p class="chat-header-sub">Ask anything about the FIA 2026 Formula 1 Regulations — all six sections loaded.</p>
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Suggestion chips (shown only on empty state)
# ---------------------------------------------------------------------------
SUGGESTIONS = [
    ("Front wing limits",   "What are the dimensional limits for the 2026 front wing?"),
    ("Parc fermé rules",    "What are the parc fermé restrictions and when do they apply?"),
    ("Power unit tokens",   "How many development tokens are allocated for power unit development?"),
    ("Tyre allocations",    "What is the tyre allocation per driver per race weekend?"),
]

if not st.session_state.messages and not st.session_state.pending_prompt:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-state-icon">🏁</div>
        <div class="empty-state-title">Ready on the PitWall</div>
        <div class="empty-state-sub">
            Your F1 2026 technical consultant is standing by.<br>
            Ask about car dimensions, sporting rules, financial regs, or operational procedures.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    for i, (label, prompt_text) in enumerate(SUGGESTIONS):
        col = col1 if i % 2 == 0 else col2
        with col:
            if st.button(f"💬 {label}", key=f"suggestion_{i}"):
                st.session_state.pending_prompt = prompt_text
                st.rerun()

else:
    # Render full Gemini-style history before streaming new messages
    for msg in st.session_state.messages:
        avatar = "🏎️" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# process_prompt — stream a user/AI exchange below the existing history
# ---------------------------------------------------------------------------
def process_prompt(prompt: str):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏎️"):
        placeholder    = st.empty()
        full_response  = ""

        if not is_online:
            placeholder.error("⚠️ Cannot reach the PitWall backend. Is `uvicorn backend.api:app` running?")
            return

        payload = {"session_id": st.session_state.session_id, "message": prompt}
        try:
            with requests.post(CHAT_ENDPOINT, json=payload, stream=True, timeout=120) as resp:
                if resp.status_code == 200:
                    for raw_line in resp.iter_lines():
                        if not raw_line:
                            continue
                        decoded = raw_line.decode("utf-8")
                        if not decoded.startswith("data: "):
                            continue
                        try:
                            data = json.loads(decoded[6:])
                            if "error" in data:
                                placeholder.error(f"Backend error: {data['error']}")
                                return
                            if data.get("done"):
                                break
                            full_response += data.get("chunk", "")
                            placeholder.markdown(full_response + "▌")
                        except json.JSONDecodeError:
                            continue

                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    placeholder.error(f"Server error {resp.status_code}: {resp.text}")

        except requests.exceptions.ConnectionError:
            placeholder.error("⚠️ Connection refused. Start backend: `uvicorn backend.api:app --reload`")
        except requests.exceptions.Timeout:
            placeholder.error("⚠️ Request timed out. The model may be slow — try again.")
        except Exception as e:
            placeholder.error(f"Unexpected error: {e}")


# ---------------------------------------------------------------------------
# Chat input — set pending + rerun so history renders before streaming starts
# ---------------------------------------------------------------------------
if user_input := st.chat_input("Ask a regulation question…", key="chat_input"):
    st.session_state.pending_prompt = user_input
    st.rerun()

# Fire any pending prompt after history is fully rendered above
if st.session_state.pending_prompt:
    prompt_to_run                   = st.session_state.pending_prompt
    st.session_state.pending_prompt = None
    process_prompt(prompt_to_run)
