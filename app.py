import streamlit as st
import os
import json
import faiss
import numpy as np
import time
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from google import genai

# --------------------
# ENV & CONFIG
# --------------------
load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")

gemini_client = genai.Client(api_key=GEMINI_KEY)
tavily = TavilyClient(api_key=TAVILY_KEY)

# --------------------
# DEMO USERS
# --------------------
USERS = {
    "admin": {"password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "Admin", "avatar": "👑"},
    "hr": {"password": hashlib.sha256("hr2024".encode()).hexdigest(), "role": "HR Manager", "avatar": "🧑‍💼"},
    "employee": {"password": hashlib.sha256("emp123".encode()).hexdigest(), "role": "Employee", "avatar": "👤"},
}

# --------------------
# LANGUAGES
# --------------------
TRANSLATIONS = {
    "English": {
        "title": "AI HR Assistant",
        "subtitle": "Your intelligent HR companion — powered by hybrid AI",
        "ask_placeholder": "Ask me anything about HR policies, leave, payroll...",
        "send_btn": "Send",
        "clear_btn": "Clear Chat",
        "thinking": "Thinking...",
        "searching_web": "Searching the web...",
        "source_kb": "📚 Knowledge Base",
        "source_web": "🌐 Web Search",
        "similarity": "Similarity",
        "login_title": "Sign In",
        "username": "Username",
        "password": "Password",
        "login_btn": "Login",
        "logout_btn": "Logout",
        "wrong_creds": "Invalid credentials. Try: admin/admin123",
        "welcome": "Welcome back",
        "you": "You",
        "assistant": "Assistant",
        "model_select": "AI Model",
        "theme_select": "Theme",
        "lang_select": "Language",
        "history": "Chat History",
        "no_history": "No messages yet. Start chatting!",
        "settings": "Settings",
    },
    "Nepali (नेपाली)": {
        "title": "AI HR सहायक",
        "subtitle": "तपाईंको बुद्धिमान HR साथी — हाइब्रिड AI द्वारा संचालित",
        "ask_placeholder": "HR नीति, बिदा, पेरोल बारे सोध्नुहोस्...",
        "send_btn": "पठाउनुहोस्",
        "clear_btn": "च्याट खाली गर्नुहोस्",
        "thinking": "सोच्दैछ...",
        "searching_web": "वेब खोजी गर्दैछ...",
        "source_kb": "📚 ज्ञान आधार",
        "source_web": "🌐 वेब खोज",
        "similarity": "समानता",
        "login_title": "साइन इन",
        "username": "प्रयोगकर्ता नाम",
        "password": "पासवर्ड",
        "login_btn": "लगिन",
        "logout_btn": "लगआउट",
        "wrong_creds": "गलत प्रमाणपत्र। प्रयास गर्नुहोस्: admin/admin123",
        "welcome": "स्वागत छ",
        "you": "तपाईं",
        "assistant": "सहायक",
        "model_select": "AI मोडेल",
        "theme_select": "थिम",
        "lang_select": "भाषा",
        "history": "च्याट इतिहास",
        "no_history": "अझै कुनै सन्देश छैन। च्याट सुरु गर्नुहोस्!",
        "settings": "सेटिङहरू",
    },
    "Hindi (हिंदी)": {
        "title": "AI HR सहायक",
        "subtitle": "आपका बुद्धिमान HR साथी — हाइब्रिड AI द्वारा संचालित",
        "ask_placeholder": "HR नीतियों, छुट्टी, वेतन के बारे में पूछें...",
        "send_btn": "भेजें",
        "clear_btn": "चैट साफ़ करें",
        "thinking": "सोच रहा हूं...",
        "searching_web": "वेब खोज रहा हूं...",
        "source_kb": "📚 ज्ञान आधार",
        "source_web": "🌐 वेब खोज",
        "similarity": "समानता",
        "login_title": "साइन इन",
        "username": "उपयोगकर्ता नाम",
        "password": "पासवर्ड",
        "login_btn": "लॉगिन",
        "logout_btn": "लॉगआउट",
        "wrong_creds": "गलत क्रेडेंशियल। प्रयास करें: admin/admin123",
        "welcome": "वापस स्वागत है",
        "you": "आप",
        "assistant": "सहायक",
        "model_select": "AI मॉडल",
        "theme_select": "थीम",
        "lang_select": "भाषा",
        "history": "चैट इतिहास",
        "no_history": "अभी कोई संदेश नहीं। चैट शुरू करें!",
        "settings": "सेटिंग्स",
    },
    "Spanish (Español)": {
        "title": "Asistente de RR.HH.",
        "subtitle": "Tu compañero inteligente de RR.HH. — impulsado por IA híbrida",
        "ask_placeholder": "Pregunta sobre políticas de RR.HH., vacaciones, nómina...",
        "send_btn": "Enviar",
        "clear_btn": "Limpiar chat",
        "thinking": "Pensando...",
        "searching_web": "Buscando en la web...",
        "source_kb": "📚 Base de conocimiento",
        "source_web": "🌐 Búsqueda web",
        "similarity": "Similitud",
        "login_title": "Iniciar sesión",
        "username": "Usuario",
        "password": "Contraseña",
        "login_btn": "Entrar",
        "logout_btn": "Salir",
        "wrong_creds": "Credenciales inválidas. Prueba: admin/admin123",
        "welcome": "Bienvenido de nuevo",
        "you": "Tú",
        "assistant": "Asistente",
        "model_select": "Modelo IA",
        "theme_select": "Tema",
        "lang_select": "Idioma",
        "history": "Historial",
        "no_history": "Sin mensajes. ¡Empieza a chatear!",
        "settings": "Configuración",
    },
    "French (Français)": {
        "title": "Assistant RH IA",
        "subtitle": "Votre compagnon RH intelligent — propulsé par l'IA hybride",
        "ask_placeholder": "Posez des questions sur les politiques RH, les congés, la paie...",
        "send_btn": "Envoyer",
        "clear_btn": "Effacer le chat",
        "thinking": "Réflexion...",
        "searching_web": "Recherche sur le web...",
        "source_kb": "📚 Base de connaissances",
        "source_web": "🌐 Recherche web",
        "similarity": "Similarité",
        "login_title": "Se connecter",
        "username": "Nom d'utilisateur",
        "password": "Mot de passe",
        "login_btn": "Connexion",
        "logout_btn": "Déconnexion",
        "wrong_creds": "Identifiants invalides. Essayez: admin/admin123",
        "welcome": "Content de vous revoir",
        "you": "Vous",
        "assistant": "Assistant",
        "model_select": "Modèle IA",
        "theme_select": "Thème",
        "lang_select": "Langue",
        "history": "Historique",
        "no_history": "Pas encore de messages. Commencez à chatter!",
        "settings": "Paramètres",
    },
}

# --------------------
# THEMES
# --------------------
THEMES = {
    "🌅 Aurora Light": {
        "bg": "#F8F6FF",
        "sidebar_bg": "#FFFFFF",
        "card_bg": "#FFFFFF",
        "primary": "#7C3AED",
        "secondary": "#A78BFA",
        "accent": "#F59E0B",
        "text": "#1E1B4B",
        "text_muted": "#6B7280",
        "user_bubble": "linear-gradient(135deg, #7C3AED, #A78BFA)",
        "bot_bubble": "#F3F0FF",
        "bot_text": "#1E1B4B",
        "border": "#E5E7EB",
        "input_bg": "#FFFFFF",
        "shadow": "rgba(124, 58, 237, 0.15)",
        "glow": "#7C3AED",
        "badge_kb": "#D1FAE5",
        "badge_web": "#DBEAFE",
        "badge_kb_text": "#065F46",
        "badge_web_text": "#1E40AF",
        "header_gradient": "linear-gradient(135deg, #7C3AED 0%, #A78BFA 50%, #F59E0B 100%)",
    },
    "🌊 Ocean Breeze": {
        "bg": "#F0F9FF",
        "sidebar_bg": "#FFFFFF",
        "card_bg": "#FFFFFF",
        "primary": "#0284C7",
        "secondary": "#38BDF8",
        "accent": "#06B6D4",
        "text": "#0C4A6E",
        "text_muted": "#64748B",
        "user_bubble": "linear-gradient(135deg, #0284C7, #38BDF8)",
        "bot_bubble": "#E0F2FE",
        "bot_text": "#0C4A6E",
        "border": "#BAE6FD",
        "input_bg": "#FFFFFF",
        "shadow": "rgba(2, 132, 199, 0.15)",
        "glow": "#0284C7",
        "badge_kb": "#D1FAE5",
        "badge_web": "#FEF3C7",
        "badge_kb_text": "#065F46",
        "badge_web_text": "#92400E",
        "header_gradient": "linear-gradient(135deg, #0284C7 0%, #06B6D4 50%, #38BDF8 100%)",
    },
    "🌿 Forest Mint": {
        "bg": "#F0FDF4",
        "sidebar_bg": "#FFFFFF",
        "card_bg": "#FFFFFF",
        "primary": "#059669",
        "secondary": "#34D399",
        "accent": "#F59E0B",
        "text": "#064E3B",
        "text_muted": "#6B7280",
        "user_bubble": "linear-gradient(135deg, #059669, #34D399)",
        "bot_bubble": "#DCFCE7",
        "bot_text": "#064E3B",
        "border": "#A7F3D0",
        "input_bg": "#FFFFFF",
        "shadow": "rgba(5, 150, 105, 0.15)",
        "glow": "#059669",
        "badge_kb": "#D1FAE5",
        "badge_web": "#DBEAFE",
        "badge_kb_text": "#065F46",
        "badge_web_text": "#1E40AF",
        "header_gradient": "linear-gradient(135deg, #059669 0%, #34D399 50%, #F59E0B 100%)",
    },
    "🌸 Cherry Blossom": {
        "bg": "#FFF1F2",
        "sidebar_bg": "#FFFFFF",
        "card_bg": "#FFFFFF",
        "primary": "#E11D48",
        "secondary": "#FB7185",
        "accent": "#F43F5E",
        "text": "#4C0519",
        "text_muted": "#9F1239",
        "user_bubble": "linear-gradient(135deg, #E11D48, #FB7185)",
        "bot_bubble": "#FFE4E6",
        "bot_text": "#4C0519",
        "border": "#FECDD3",
        "input_bg": "#FFFFFF",
        "shadow": "rgba(225, 29, 72, 0.15)",
        "glow": "#E11D48",
        "badge_kb": "#D1FAE5",
        "badge_web": "#EDE9FE",
        "badge_kb_text": "#065F46",
        "badge_web_text": "#4C1D95",
        "header_gradient": "linear-gradient(135deg, #E11D48 0%, #FB7185 50%, #FFF1F2 100%)",
    },
    "🌙 Midnight Dark": {
        "bg": "#0F0F1A",
        "sidebar_bg": "#1A1A2E",
        "card_bg": "#16213E",
        "primary": "#8B5CF6",
        "secondary": "#A78BFA",
        "accent": "#F59E0B",
        "text": "#E2E8F0",
        "text_muted": "#94A3B8",
        "user_bubble": "linear-gradient(135deg, #8B5CF6, #A78BFA)",
        "bot_bubble": "#1E293B",
        "bot_text": "#E2E8F0",
        "border": "#334155",
        "input_bg": "#1E293B",
        "shadow": "rgba(139, 92, 246, 0.3)",
        "glow": "#8B5CF6",
        "badge_kb": "#064E3B",
        "badge_web": "#1E3A5F",
        "badge_kb_text": "#6EE7B7",
        "badge_web_text": "#93C5FD",
        "header_gradient": "linear-gradient(135deg, #8B5CF6 0%, #6366F1 50%, #A78BFA 100%)",
    },
    "🔥 Sunset Ember": {
        "bg": "#FFFBEB",
        "sidebar_bg": "#FFFFFF",
        "card_bg": "#FFFFFF",
        "primary": "#D97706",
        "secondary": "#F59E0B",
        "accent": "#EF4444",
        "text": "#451A03",
        "text_muted": "#78350F",
        "user_bubble": "linear-gradient(135deg, #D97706, #F59E0B)",
        "bot_bubble": "#FEF3C7",
        "bot_text": "#451A03",
        "border": "#FDE68A",
        "input_bg": "#FFFFFF",
        "shadow": "rgba(217, 119, 6, 0.15)",
        "glow": "#D97706",
        "badge_kb": "#DCFCE7",
        "badge_web": "#EDE9FE",
        "badge_kb_text": "#166534",
        "badge_web_text": "#4C1D95",
        "header_gradient": "linear-gradient(135deg, #D97706 0%, #F59E0B 50%, #FCD34D 100%)",
    },
}

AI_MODELS = {
    "⚡ Gemini 2.5 Flash (Fast)": "gemini-2.5-flash",
    "🧠 Gemini 2.5 Pro (Smart)": "gemini-2.5-pro",
    "🌟 Gemini 1.5 Flash": "gemini-1.5-flash",
    "💎 Gemini 1.5 Pro": "gemini-1.5-pro",
}

# --------------------
# PAGE CONFIG
# --------------------
st.set_page_config(
    page_title="AI HR Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------
# SESSION STATE INIT
# --------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "messages" not in st.session_state:
    st.session_state.messages = []
if "theme" not in st.session_state:
    st.session_state.theme = "🌅 Aurora Light"
if "language" not in st.session_state:
    st.session_state.language = "English"
if "model" not in st.session_state:
    st.session_state.model = "⚡ Gemini 2.5 Flash (Fast)"


def t(key):
    lang = st.session_state.language
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, key)


def get_theme():
    return THEMES[st.session_state.theme]


def inject_css(theme):
    bg = theme["bg"]
    sidebar_bg = theme["sidebar_bg"]
    card_bg = theme["card_bg"]
    primary = theme["primary"]
    secondary = theme["secondary"]
    accent = theme["accent"]
    text = theme["text"]
    text_muted = theme["text_muted"]
    user_bubble = theme["user_bubble"]
    bot_bubble = theme["bot_bubble"]
    bot_text = theme["bot_text"]
    border = theme["border"]
    input_bg = theme["input_bg"]
    shadow = theme["shadow"]
    glow = theme["glow"]
    badge_kb = theme["badge_kb"]
    badge_web = theme["badge_web"]
    badge_kb_text = theme["badge_kb_text"]
    badge_web_text = theme["badge_web_text"]
    header_gradient = theme["header_gradient"]

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

* {{
    font-family: 'Inter', sans-serif;
    box-sizing: border-box;
}}

/* ── ROOT & APP BG ── */
.stApp {{
    background: {bg};
    color: {text};
}}

/* ── HIDE STREAMLIT DEFAULTS ── */
#MainMenu, footer, header {{visibility: hidden;}}
.stDeployButton {{display: none;}}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {{
    background: {sidebar_bg} !important;
    border-right: 1px solid {border};
    box-shadow: 4px 0 24px {shadow};
}}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p {{
    color: {text} !important;
}}

/* ── HERO HEADER ── */
.hero-header {{
    background: {header_gradient};
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 40px {shadow};
    position: relative;
    overflow: hidden;
    animation: fadeInDown 0.6s ease;
}}
.hero-header::before {{
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}}
.hero-header::after {{
    content: '';
    position: absolute;
    bottom: -30%;
    left: 5%;
    width: 200px;
    height: 200px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
    animation: float 8s ease-in-out infinite reverse;
}}
.hero-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: white;
    margin: 0 0 8px;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}}
.hero-subtitle {{
    font-size: 1rem;
    color: rgba(255,255,255,0.85);
    margin: 0;
    position: relative;
    z-index: 1;
}}
.hero-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.2);
    color: white;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
    backdrop-filter: blur(10px);
    position: relative;
    z-index: 1;
}}

/* ── CHAT CONTAINER ── */
.chat-container {{
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 8px 0;
}}

/* ── MESSAGE BUBBLES ── */
.message-row {{
    display: flex;
    align-items: flex-end;
    gap: 12px;
    animation: slideIn 0.35s ease;
}}
.message-row.user {{
    flex-direction: row-reverse;
}}
.avatar {{
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    border: 2px solid {border};
}}
.avatar.user-av {{
    background: {user_bubble};
    border-color: transparent;
}}
.avatar.bot-av {{
    background: {bot_bubble};
}}
.bubble {{
    max-width: 72%;
    padding: 14px 18px;
    border-radius: 18px;
    line-height: 1.6;
    font-size: 0.95rem;
    box-shadow: 0 2px 12px {shadow};
    position: relative;
}}
.bubble.user-bubble {{
    background: {user_bubble};
    color: white;
    border-bottom-right-radius: 4px;
}}
.bubble.bot-bubble {{
    background: {bot_bubble};
    color: {bot_text};
    border-bottom-left-radius: 4px;
    border: 1px solid {border};
}}
.msg-meta {{
    font-size: 0.72rem;
    color: {text_muted};
    margin-top: 5px;
    padding: 0 6px;
}}
.msg-meta.user {{ text-align: right; }}

/* ── SOURCE BADGE ── */
.source-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-top: 8px;
}}
.source-badge.kb {{
    background: {badge_kb};
    color: {badge_kb_text};
}}
.source-badge.web {{
    background: {badge_web};
    color: {badge_web_text};
}}

/* ── INPUT AREA ── */
.input-wrapper {{
    background: {card_bg};
    border: 1.5px solid {border};
    border-radius: 16px;
    padding: 4px;
    box-shadow: 0 4px 20px {shadow};
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 20px;
    transition: border-color 0.2s, box-shadow 0.2s;
}}
.input-wrapper:focus-within {{
    border-color: {primary};
    box-shadow: 0 4px 24px {shadow}, 0 0 0 3px {glow}22;
}}

/* ── STREAMLIT WIDGET OVERRIDES ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background: {input_bg} !important;
    color: {text} !important;
    border: 1.5px solid {border} !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: {primary} !important;
    box-shadow: 0 0 0 3px {glow}22 !important;
}}

/* ── BUTTONS ── */
.stButton > button {{
    background: {user_bubble} !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 15px {shadow} !important;
    cursor: pointer !important;
}}
.stButton > button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px {shadow} !important;
    filter: brightness(1.07) !important;
}}
.stButton > button:active {{
    transform: translateY(0) !important;
}}

/* ── SELECTBOX ── */
.stSelectbox > div > div {{
    background: {input_bg} !important;
    border: 1.5px solid {border} !important;
    border-radius: 12px !important;
    color: {text} !important;
}}

/* ── DIVIDER ── */
hr {{
    border-color: {border} !important;
    margin: 20px 0 !important;
}}

/* ── SIDEBAR USER CARD ── */
.user-card {{
    background: linear-gradient(135deg, {primary}18, {secondary}18);
    border: 1px solid {border};
    border-radius: 16px;
    padding: 16px;
    margin-bottom: 20px;
    text-align: center;
}}
.user-card .user-avatar {{
    font-size: 2.5rem;
    margin-bottom: 8px;
}}
.user-card .user-name {{
    font-weight: 700;
    font-size: 1rem;
    color: {text};
}}
.user-card .user-role {{
    font-size: 0.78rem;
    color: {text_muted};
    background: {primary}20;
    padding: 2px 10px;
    border-radius: 10px;
    display: inline-block;
    margin-top: 4px;
}}

/* ── STAT CARDS ── */
.stat-card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 2px 10px {shadow};
}}
.stat-icon {{
    font-size: 1.5rem;
}}
.stat-info .stat-label {{
    font-size: 0.72rem;
    color: {text_muted};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.stat-info .stat-value {{
    font-size: 1.1rem;
    font-weight: 700;
    color: {text};
}}

/* ── LOGIN CARD ── */
.login-card {{
    background: {card_bg};
    border: 1px solid {border};
    border-radius: 24px;
    padding: 40px;
    box-shadow: 0 20px 60px {shadow};
    max-width: 420px;
    margin: 0 auto;
    animation: fadeInUp 0.5s ease;
}}
.login-logo {{
    font-size: 3.5rem;
    text-align: center;
    margin-bottom: 8px;
    animation: pulse 2s infinite;
}}
.login-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: {text};
    text-align: center;
    margin-bottom: 4px;
}}
.login-hint {{
    font-size: 0.8rem;
    color: {text_muted};
    text-align: center;
    background: {primary}10;
    padding: 8px 16px;
    border-radius: 10px;
    margin: 16px 0;
    border: 1px solid {primary}22;
}}

/* ── TYPING INDICATOR ── */
.typing-indicator {{
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 12px 16px;
    background: {bot_bubble};
    border: 1px solid {border};
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    width: fit-content;
}}
.typing-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: {primary};
    animation: typingBounce 1.2s infinite;
}}
.typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
.typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}

/* ── SIMILARITY BAR ── */
.sim-bar-wrapper {{
    margin-top: 6px;
}}
.sim-label {{
    font-size: 0.72rem;
    color: {text_muted};
    margin-bottom: 3px;
}}
.sim-bar-bg {{
    height: 5px;
    background: {border};
    border-radius: 4px;
    overflow: hidden;
}}
.sim-bar-fill {{
    height: 100%;
    background: {user_bubble};
    border-radius: 4px;
    transition: width 1s ease;
}}

/* ── SECTION HEADING ── */
.section-heading {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: {text_muted};
    margin: 20px 0 12px;
}}

/* ── QUICK CHIPS ── */
.chip-grid {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 16px;
}}
.chip {{
    background: {primary}15;
    color: {primary};
    border: 1px solid {primary}30;
    border-radius: 20px;
    padding: 5px 13px;
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}}
.chip:hover {{
    background: {primary}25;
    transform: translateY(-1px);
}}

/* ── ANIMATIONS ── */
@keyframes fadeInDown {{
    from {{ opacity: 0; transform: translateY(-20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(20px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}
@keyframes slideIn {{
    from {{ opacity: 0; transform: translateX(-10px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes float {{
    0%, 100% {{ transform: translateY(0) scale(1); }}
    50% {{ transform: translateY(-20px) scale(1.05); }}
}}
@keyframes pulse {{
    0%, 100% {{ transform: scale(1); }}
    50% {{ transform: scale(1.08); }}
}}
@keyframes typingBounce {{
    0%, 60%, 100% {{ transform: translateY(0); opacity: 0.4; }}
    30% {{ transform: translateY(-8px); opacity: 1; }}
}}
@keyframes shimmer {{
    0% {{ background-position: -200% center; }}
    100% {{ background-position: 200% center; }}
}}

/* ── SCROLLBAR ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {border}; border-radius: 10px; }}
::-webkit-scrollbar-thumb:hover {{ background: {primary}60; }}

/* ── SIDEBAR SECTION LABEL ── */
.sidebar-section {{
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: {text_muted};
    margin: 16px 0 8px;
    padding-bottom: 4px;
    border-bottom: 1px solid {border};
}}

/* ── EMPTY STATE ── */
.empty-state {{
    text-align: center;
    padding: 60px 20px;
    color: {text_muted};
}}
.empty-state-icon {{
    font-size: 4rem;
    margin-bottom: 16px;
    animation: float 4s ease-in-out infinite;
}}
.empty-state-text {{
    font-size: 1.1rem;
    font-weight: 500;
    color: {text};
    margin-bottom: 8px;
}}
.empty-state-sub {{
    font-size: 0.85rem;
    color: {text_muted};
}}
</style>
""", unsafe_allow_html=True)


# --------------------
# EMBEDDING & INDEX (cached)
# --------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("BAAI/bge-small-en-v1.5")


embedder = load_model()


@st.cache_resource
def load_data():
    with open("data.json", "r") as f:
        data = json.load(f)
    docs = []
    for item in data:
        docs.append(f"Question:\n{item['question']}\n\nAnswer:\n{item['answer']}")
    return docs


documents = load_data()


@st.cache_resource
def create_index(_documents):
    vectors = embedder.encode(_documents, normalize_embeddings=True)
    vectors = np.array(vectors).astype("float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


index = create_index(documents)


# --------------------
# SEARCH FUNCTIONS
# --------------------
def search_knowledge(query):
    q = embedder.encode([query], normalize_embeddings=True)
    q = np.array(q).astype("float32")
    score, idx = index.search(q, 1)
    return score[0][0], documents[idx[0][0]]


def web_search(query):
    result = tavily.search(query=query, max_results=3)
    content = ""
    for item in result["results"]:
        content += item["content"] + "\n\n"
    return content


def generate_answer(prompt, model_key):
    model_id = AI_MODELS[model_key]
    response = gemini_client.models.generate_content(model=model_id, contents=prompt)
    return response.text


# --------------------
# LOGIN PAGE
# --------------------
def login_page():
    theme = get_theme()
    inject_css(theme)

    col1, col2, col3 = st.columns([1, 1.4, 1])
    with col2:
        st.markdown(f"""
        <div class="login-card">
            <div class="login-logo">🤖</div>
            <div class="login-title">AI HR Assistant</div>
            <p style="text-align:center;color:{theme['text_muted']};font-size:0.88rem;margin-top:4px;">
                Intelligent HR powered by hybrid AI
            </p>
            <div class="login-hint">
                Demo: <strong>admin</strong> / <strong>admin123</strong><br>
                or <strong>hr</strong> / <strong>hr2024</strong>&nbsp;&nbsp;·&nbsp;&nbsp;<strong>employee</strong> / <strong>emp123</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.container():
            username = st.text_input("👤 Username", placeholder="Enter username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter password")

            col_a, col_b = st.columns(2)
            with col_a:
                theme_sel = st.selectbox("🎨 Theme", list(THEMES.keys()), key="login_theme")
            with col_b:
                lang_sel = st.selectbox("🌐 Language", list(TRANSLATIONS.keys()), key="login_lang")

            if st.button("🚀 Sign In", use_container_width=True):
                if username in USERS:
                    hashed = hashlib.sha256(password.encode()).hexdigest()
                    if USERS[username]["password"] == hashed:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.theme = theme_sel
                        st.session_state.language = lang_sel
                        st.rerun()
                    else:
                        st.error("❌ Wrong password.")
                else:
                    st.error("❌ User not found.")


# --------------------
# MAIN CHAT APP
# --------------------
def chat_app():
    theme = get_theme()
    inject_css(theme)
    T = lambda k: t(k)
    user_info = USERS[st.session_state.username]

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div class="user-card">
            <div class="user-avatar">{user_info['avatar']}</div>
            <div class="user-name">{st.session_state.username.title()}</div>
            <span class="user-role">{user_info['role']}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="sidebar-section">{T("settings")}</div>', unsafe_allow_html=True)

        st.session_state.theme = st.selectbox(
            T("theme_select"), list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.theme)
        )
        st.session_state.language = st.selectbox(
            T("lang_select"), list(TRANSLATIONS.keys()),
            index=list(TRANSLATIONS.keys()).index(st.session_state.language)
        )
        st.session_state.model = st.selectbox(
            T("model_select"), list(AI_MODELS.keys()),
            index=list(AI_MODELS.keys()).index(st.session_state.model)
        )

        st.markdown("---")
        st.markdown(f'<div class="sidebar-section">Stats</div>', unsafe_allow_html=True)

        n_msgs = len(st.session_state.messages)
        n_user = sum(1 for m in st.session_state.messages if m["role"] == "user")
        n_kb = sum(1 for m in st.session_state.messages if m.get("source") == "kb")
        n_web = sum(1 for m in st.session_state.messages if m.get("source") == "web")

        st.markdown(f"""
        <div class="stat-card">
            <span class="stat-icon">💬</span>
            <div class="stat-info">
                <div class="stat-label">Total Messages</div>
                <div class="stat-value">{n_msgs}</div>
            </div>
        </div>
        <div class="stat-card">
            <span class="stat-icon">📚</span>
            <div class="stat-info">
                <div class="stat-label">KB Answers</div>
                <div class="stat-value">{n_kb}</div>
            </div>
        </div>
        <div class="stat-card">
            <span class="stat-icon">🌐</span>
            <div class="stat-info">
                <div class="stat-label">Web Searches</div>
                <div class="stat-value">{n_web}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        col_c, col_d = st.columns(2)
        with col_c:
            if st.button(f"🗑️ {T('clear_btn')}", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        with col_d:
            if st.button(f"🚪 {T('logout_btn')}", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.messages = []
                st.rerun()

    # ── MAIN AREA ──────────────────────────────
    st.markdown(f"""
    <div class="hero-header">
        <div class="hero-badge">✨ Hybrid AI · Multilingual · Real-time Search</div>
        <div class="hero-title">🤖 {T('title')}</div>
        <div class="hero-subtitle">{T('subtitle')}</div>
    </div>
    """, unsafe_allow_html=True)

    # Quick Chips
    quick_questions = [
        "What is the leave policy?",
        "How do I apply for leave?",
        "What are working hours?",
        "How to claim reimbursements?",
        "What is probation period?",
        "How does payroll work?",
    ]
    chip_html = '<div class="chip-grid">' + "".join(
        f'<span class="chip">{q}</span>' for q in quick_questions
    ) + "</div>"
    st.markdown(chip_html, unsafe_allow_html=True)

    # Chat History
    if not st.session_state.messages:
        st.markdown(f"""
        <div class="empty-state">
            <div class="empty-state-icon">💬</div>
            <div class="empty-state-text">{T('no_history')}</div>
            <div class="empty-state-sub">Try clicking one of the quick questions above!</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            time_str = msg.get("time", "")
            source = msg.get("source", "")
            sim = msg.get("similarity", None)

            if role == "user":
                st.markdown(f"""
                <div class="message-row user">
                    <div class="avatar user-av">{user_info['avatar']}</div>
                    <div style="flex:1;display:flex;flex-direction:column;align-items:flex-end;">
                        <div class="bubble user-bubble">{content}</div>
                        <div class="msg-meta user">{T('you')} · {time_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                source_badge = ""
                if source == "kb":
                    badge_html = f'<span class="source-badge kb">{T("source_kb")}</span>'
                else:
                    badge_html = f'<span class="source-badge web">{T("source_web")}</span>'

                sim_bar = ""
                if sim is not None:
                    pct = int(sim * 100)
                    sim_bar = f"""
                    <div class="sim-bar-wrapper">
                        <div class="sim-label">{T('similarity')}: {pct}%</div>
                        <div class="sim-bar-bg"><div class="sim-bar-fill" style="width:{pct}%"></div></div>
                    </div>"""

                st.markdown(f"""
                <div class="message-row">
                    <div class="avatar bot-av">🤖</div>
                    <div style="flex:1;">
                        <div class="bubble bot-bubble">
                            {content}
                            <div style="margin-top:10px;">{badge_html}</div>
                            {sim_bar}
                        </div>
                        <div class="msg-meta">{T('assistant')} · {time_str}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Input
    st.markdown("---")
    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        question = st.text_input(
            label="Question Input",
            placeholder=T("ask_placeholder"),
            label_visibility="collapsed",
            key="question_input"
        )
    with col_btn:
        send = st.button(f"➤ {T('send_btn')}", use_container_width=True)

    if send and question.strip():
        now = datetime.now().strftime("%H:%M")

        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "time": now,
        })

        # Hybrid search
        score, context = search_knowledge(question)

        if score > 0.65:
            source = "kb"
            final_context = context
        else:
            source = "web"
            with st.spinner(T("searching_web")):
                final_context = web_search(question)

        # Language instruction
        lang = st.session_state.language
        lang_note = f"Respond in {lang.split(' ')[0]} language." if lang != "English" else ""

        prompt = f"""You are a professional HR AI assistant. Answer the user's question using ONLY the context provided.
If the context does not contain the answer, say you don't know.
Be concise, helpful, and professional. {lang_note}

Context:
{final_context}

Question:
{question}

Answer:"""

        with st.spinner(T("thinking")):
            answer = generate_answer(prompt, st.session_state.model)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "time": now,
            "source": source,
            "similarity": float(score) if source == "kb" else None,
        })

        st.rerun()


# --------------------
# MAIN ROUTER
# --------------------
if not st.session_state.logged_in:
    login_page()
else:
    chat_app()