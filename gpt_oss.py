import os
import re
import time
import hashlib
import random
import shutil
import math
from typing import List, Dict, Any, Tuple
import concurrent.futures as _f

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import numpy as np

# --- Web UI / App ---
st.set_page_config(page_title="로컬 RAG 챗봇 (GPT-OSS + Chroma)", page_icon="🧠", layout="wide")

# --- GPT-OSS ---
from gpt_oss import GPTModel
from gpt_oss.embeddings import GPTEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===================== 환경 로드 =====================
load_dotenv()

EMB_BACKEND = os.getenv("EMB_BACKEND", "gpt-oss").lower()
GPT_BASE_URL = os.getenv("GPT_BASE_URL", "http://localhost:8080")  # 로컬 서버 URL로 변경
GPT_GEN_MODEL = os.getenv("GPT_GEN_MODEL", "gpt-oss-20b")  # 모델 이름 수정 (예: 20b 모델)
GPT_EMBED_MODEL = os.getenv("GPT_EMBED_MODEL", "gpt-oss-embeddings")

CSV_DEFAULT = os.getenv("CSV_DEFAULT", "test.csv")
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_creation")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "")

CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))
MAX_EMBED_CHARS = int(os.getenv("MAX_EMBED_CHARS", "3500"))
FRESH = os.getenv("FRESH", "true").lower() in ("1", "true", "yes", "y")

# ===================== 공용 캐시/리소스 =====================
@st.cache_resource(show_spinner=False)
def _get_emb() -> GPTEmbeddings:
    return GPTEmbeddings(model=GPT_EMBED_MODEL, base_url=GPT_BASE_URL)

@st.cache_resource(show_spinner=False)
def _get_llm(temp: float = 0.2, num_predict: int = 200) -> GPTModel:
    return GPTModel(
        model_name=GPT_GEN_MODEL,
        temperature=temp,
        max_tokens=num_predict,
        api_url=GPT_BASE_URL
    )

# ===================== 유틸 =====================
@st.cache_data(show_spinner=False)
def _cached_read_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    return df

def df_fingerprint(df: pd.DataFrame) -> str:
    parts = [(row.get("title", "") or "") + (row.get("content", "") or "") for _, row in df.iterrows()]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

def persist_path(persist_dir: str, fp: str) -> Tuple[str, str]:
    d = os.path.join(persist_dir, f"chroma_{fp[:12]}")
    return d, f"creation_{fp[:12]}"

def load_csv(csv_path: str) -> pd.DataFrame:
    csv_path = csv_path if os.path.isabs(csv_path) else os.path.join(os.getcwd(), csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    df = _cached_read_csv(csv_path)
    need_cols = {"url", "title", "content", "references", "further_refs"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼이 누락되었습니다: {missing}")
    return df

def need_force_fetch(url: str) -> bool:
    try:
        host = re.sub(r"^https?://", "", url).split("/")[0]
        return any(host.endswith(dom) for dom in FORCE_FETCH_DOMAINS)
    except Exception:
        return False

def _smart_select_main(soup: BeautifulSoup):
    for css in [
        "article", ".fr-view", ".rd-content", ".board_view", ".boardView", ".content",
        "#content", "#article", "#view", ".editor_content", ".xe_content", ".se-component",
    ]:
        node = soup.select_one(css)
        if node and node.get_text(strip=True):
            return node
    return soup.body or soup

@st.cache_data(show_spinner=False)
def fetch_url_text(url: str, timeout: int = 12, max_len: int = 25000, retries: int = 2) -> str:
    if not url or not re.match(r"^https?://", url):
        return ""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (CreationKR/1.0)"}, timeout=timeout)
            r.raise_for_status()
            if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
                r.encoding = r.apparent_encoding or "utf-8"
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            main = _smart_select_main(soup)
            text = (main or soup).get_text("\n")
            lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
            text = "\n".join([ln for ln in lines if ln])[:max_len]
            if len(text) < 150 and attempt < retries:
                time.sleep(0.2)
                continue
            return text
        except Exception:
            time.sleep(0.2)
    return ""

def safe_truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars]
    last = max(cut.rfind("\n"), cut.rfind(". "), cut.rfind("。"), cut.rfind("! "), cut.rfind("? "))
    if last >= max_chars * 0.7:
        return cut[:last].rstrip()
    return cut.rstrip()

@st.cache_data(show_spinner=False)
def _split_base_text(base_text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(base_text)

# 기타 필요한 함수들 (검색, 인덱스 구축 등)들은 위와 비슷한 방식으로 GPT-OSS와 호환되게 수정 가능합니다.

# ===================== 프롬프트 & RAG 체인 =====================
SYSTEM_PROMPT = (
    "당신은 신뢰할 수 있는 한국어 연구 보조자입니다. "
    "제공된 컨텍스트를 지식의 근거로 삼되, 문장을 그대로 가져오지 말고 반드시 재구성하여 자연스럽게 설명하세요. "
    "표현 방식, 문장 구조, 설명 흐름을 바꿔서 새롭게 서술해야 합니다. "
    "컨텍스트에 없는 사실은 말하지 말고, 모르면 모른다고 답하세요."
)


USER_Q_TEMPLATE = (
    "질문: {question}\n\n"
    "[컨텍스트]\n{context}\n\n"
    "요구사항:\n"
    "- 컨텍스트에 정보가 없으면 '문서에 명시되어 있지 않습니다'라고 답하세요."
    "- 본문에는 링크/출처를 쓰지 마세요.\n"
    "- 잘못된 내용을 지어내지 마세요.\n"
    "- 맞춤법은 지키세요.\n"
)

# ===================== UI =====================
st.title("🤖 창조 과학 챗봇")

# GPT-OSS 로컬 서버 워밍(첫 질의 딜레이 완화)
llm = _get_llm()
_ = llm.generate("Hello, how are you?")

# 세션 상태
if "history" not in st.session_state:
    st.session_state.history = []

# 채팅 영역
st.markdown("### 💬 대화")
for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

user_msg = st.chat_input("질문을 입력하세요… (로컬 문서 기반으로 답합니다)")
if user_msg:
    st.session_state.history.append(("user", user_msg))
    with st.chat_message("user"):
        st.markdown(user_msg)

    with st.chat_message("assistant"):
        try:
            answer = llm.generate(user_msg)
            st.markdown(answer)
        except Exception as e:
            st.error(f"응답 생성 실패: {e}")

    st.session_state.history.append(("assistant", answer))
