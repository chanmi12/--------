# -*- coding: utf-8 -*-
"""
병렬 임베딩 + 완전 로컬 인덱서 (Ollama + Chroma)
"""

import os, re, time, hashlib, shutil, random
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ===================== 기본 설정 =====================
load_dotenv()

CSV_DEFAULT        = os.getenv("CSV_DEFAULT", "test.csv")
PERSIST_DIR        = os.getenv("PERSIST_DIR", "./chroma_creation")
COLLECTION_NAME    = (os.getenv("COLLECTION_NAME") or "").strip()

EMB_BACKEND        = os.getenv("EMB_BACKEND", "ollama")
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").replace("/v1","")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")

CHUNK_CHARS        = int(os.getenv("CHUNK_CHARS", 1800))
CHUNK_OVERLAP      = int(os.getenv("CHUNK_OVERLAP", 200))
MAX_EMBED_CHARS    = int(os.getenv("MAX_EMBED_CHARS", 6000))

# ===================== 임베딩 워커 =====================
def embed_worker(args):
    """병렬 프로세스에서 실행되는 임베딩 작업."""
    text, model_name, base_url = args
    emb = OllamaEmbeddings(model=model_name, base_url=base_url)
    vec = emb.embed_query(text)
    return vec


# ===================== 문서 fingerprint =====================
def df_fingerprint(df: pd.DataFrame) -> str:
    parts = [(row.get("title","") or "") + (row.get("content","") or "") for _, row in df.iterrows()]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()

def persist_path(persist_dir: str, fp: str):
    cname = COLLECTION_NAME if COLLECTION_NAME else f"creation_{fp[:12]}"
    d = os.path.join(persist_dir, f"chroma_{cname}")
    return d, cname

# ===================== CSV 로딩 =====================
def load_csv(csv_path: str):
    csv_path = csv_path if os.path.isabs(csv_path) else os.path.join(os.getcwd(), csv_path)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    need_cols = {"url","title","content","references","further_refs"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV에 필요한 컬럼 누락: {missing}")
    return df

# ===================== Chunk 생성 =====================
def docs_from_df(df):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP,
        length_function=len, separators=["\n\n", "\n", " ", ""]
    )
    docs, ids = [], []

    for ridx, row in df.iterrows():
        row_id = str(ridx)
        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        if not title and not content:
            continue

        base = f"{title}\n\n{content}".strip()
        chunks = splitter.split_text(base)

        final = []
        for ch in chunks:
            if len(ch) <= MAX_EMBED_CHARS:
                final.append(ch)
            else:
                # 자동 재분할
                start = 0
                while start < len(ch):
                    piece = ch[start:start+MAX_EMBED_CHARS]
                    final.append(piece)
                    start += len(piece)

        for cidx, chunk in enumerate(final):
            did = hashlib.sha1(chunk.encode()).hexdigest()[:16]
            did = f"{row_id}-{cidx}-{did}"
            meta = {
                "title": title,
                "row_id": row_id,
                "chunk_id": f"{row_id}-{cidx}"
            }
            docs.append(Document(page_content=chunk, metadata=meta))
            ids.append(did)

    return docs, ids


# ===================== 병렬 임베딩 → Chroma 저장 =====================
def parallel_embedding_and_insert(docs, ids, store, workers=4):
    print(f"[parallel] CPU workers = {workers}")

    args_list = [
        (doc.page_content, OLLAMA_EMBED_MODEL, OLLAMA_BASE_URL)
        for doc in docs
    ]

    vectors = [None] * len(docs)

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(embed_worker, args): idx for idx, args in enumerate(args_list)}
        for i, fut in enumerate(as_completed(futs)):
            idx = futs[fut]
            try:
                vectors[idx] = fut.result()
            except Exception as e:
                print(f"[parallel] embedding failed idx={idx}, err={e}")
                vectors[idx] = None

            if (i+1) % 20 == 0:
                print(f"[parallel] {i+1}/{len(docs)} embeddings computed")

    # ====== 저장 준비 ======
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]

    # ====== Chroma 저장 (정답) ======
    store.add_texts(
        texts=texts,
        metadatas=metadatas,
        ids=ids,
        embeddings=vectors
    )

# ===================== 메인 =====================
def main():
    print("========== Build Parallel Embedding Index ==========")

    df = load_csv(CSV_DEFAULT)
    fp = df_fingerprint(df)
    d, cname = persist_path(PERSIST_DIR, fp)

    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

    docs, ids = docs_from_df(df)
    print(f"[index] chunks = {len(docs)}")

    # store 생성
    store = Chroma(
        persist_directory=d,
        collection_name=cname,
        embedding_function=None  
    )

    # 병렬 임베딩 + 저장
    parallel_embedding_and_insert(docs, ids, store, workers=os.cpu_count())

    print("[index] ✅ done.")

if __name__ == "__main__":
    main()
