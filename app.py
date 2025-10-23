# app.py — Multi-LLM (Gemini & DeepSeek), 3 Chunking, 2 Embeddings
import os
import re
import gc
import uuid
import time
import shutil
import base64
import requests
import torch
import fitz            # PyMuPDF
import pdfplumber
import pandas as pd
import streamlit as st
from datetime import datetime
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import streamlit as st
from PIL import Image
import fitz  # PyMuPDF



# ========== Konfigurasi Aplikasi ==========
st.set_page_config(page_title="Chatbot SMKB BCMS", layout="wide")
UPLOAD_FOLDER = "./uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
VECTORSTORE_DIR = f"./chroma_{str(uuid.uuid4())}"

# ========== Konfigurasi Gemini ==========
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ========== Util ==========
def clean_response(text: str):
    """Hapus tag <think> ... </think> dan trimming."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def load_image_as_base64(path):
    with open(path, "rb") as img:
        return base64.b64encode(img.read()).decode()

# ========== Ekstraksi PDF (Text, Table, Image markers) ==========
def extract_text_table_image(pdf_path: str, merge_tables=True):
    """
    Ekstrak teks, tabel, dan gambar dari PDF.
    Jika merge_tables=True, tabel lintas halaman dengan header sama akan digabung.
    """
    documents = []
    file_name = os.path.basename(pdf_path)
    pdf_img = fitz.open(pdf_path)

    with pdfplumber.open(pdf_path) as pdf_text:
        total_pages = len(pdf_text.pages)
        progress_bar = st.progress(0, text=f"Memproses {file_name}...")

        last_table_header = None
        pending_table_rows = []

        for i, (page_text, page_img) in enumerate(zip(pdf_text.pages, pdf_img)):
            page_number = i + 1

            # --- Text
            text = page_text.extract_text()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_name, "page": page_number, "modality": "text"}
                    )
                )

            # --- Tables
            tables = page_text.extract_tables()
            if tables:
                for table in tables:
                    if not table:
                        continue

                    header = tuple(table[0])  # anggap baris pertama sebagai header
                    rows = table[1:]

                    # Jika header sama dengan halaman sebelumnya → gabungkan
                    if merge_tables and last_table_header == header:
                        pending_table_rows.extend(rows)
                    else:
                        # simpan tabel sebelumnya dulu
                        if pending_table_rows and last_table_header:
                            table_str = "\n".join(
                                [" | ".join(map(str, last_table_header))] +
                                [" | ".join(map(str, r)) for r in pending_table_rows]
                            )
                            documents.append(
                                Document(
                                    page_content=table_str,
                                    metadata={"source": file_name, "page": page_number-1, "modality": "table"}
                                )
                            )

                        # reset ke tabel baru
                        last_table_header = header
                        pending_table_rows = rows

            # --- Images
            for img_index, _ in enumerate(page_img.get_images(full=True)):
                documents.append(
                    Document(
                        page_content=f"[Gambar ditemukan pada halaman {page_number} - Gambar ke-{img_index+1}]",
                        metadata={"source": file_name, "page": page_number, "modality": "image"}
                    )
                )

            progress_bar.progress((page_number) / total_pages, text=f"Halaman {page_number}/{total_pages}")

        # simpan tabel terakhir
        if pending_table_rows and last_table_header:
            table_str = "\n".join(
                [" | ".join(map(str, last_table_header))] +
                [" | ".join(map(str, r)) for r in pending_table_rows]
            )
            documents.append(
                Document(
                    page_content=table_str,
                    metadata={"source": file_name, "page": total_pages, "modality": "table"}
                )
            )

        progress_bar.empty()
    pdf_img.close()
    return documents

# ========== Chunking Helpers ==========
def paragraph_chunk(text: str):
    # paragraf minimal 30 karakter
    return [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]

def modality_specific_chunking(all_docs):
    """Pisahkan text ke paragraf, table & image dibiarkan per entri."""
    texts = []
    for doc in all_docs:
        if doc.metadata.get("modality") == "text":
            for chunk in paragraph_chunk(doc.page_content):
                texts.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": doc.metadata.get("source", "-"),
                            "page": doc.metadata.get("page", "-"),
                            "modality": "text",
                        },
                    )
                )
        else:
            texts.append(doc)
    return texts

def recursive_chunking(all_docs, chunk_size=550, chunk_overlap=138):
    """RecursiveCharacterTextSplitter untuk semua modality text; table/image tetap utuh."""
    text_docs = [d for d in all_docs if d.metadata.get("modality") == "text"]
    other_docs = [d for d in all_docs if d.metadata.get("modality") in ("table", "image")]

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " "],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    split_texts = splitter.split_documents(text_docs)
    return split_texts + other_docs

def table_aware_chunking(all_docs):
    """
    Table-Aware sederhana:
    - Table modality tetap per tabel.
    - Text modality dipotong per paragraf, tetapi paragraf yang 'mirip tabel' (mengandung tab/|) dipisah sebagai satu blok.
    """
    texts = []
    for doc in all_docs:
        mod = doc.metadata.get("modality")
        if mod == "table":
            texts.append(doc)
        elif mod == "image":
            texts.append(doc)
        else:  # text
            # Deteksi paragraf
            paras = [p for p in doc.page_content.split("\n\n") if p.strip()]
            for p in paras:
                if ("|" in p) or ("\t" in p) or ("table" in p.lower()):
                    # treat as table-like block (keep as is but tag as table-aware)
                    texts.append(
                        Document(
                            page_content=p.strip(),
                            metadata={
                                "source": doc.metadata.get("source", "-"),
                                "page": doc.metadata.get("page", "-"),
                                "modality": "table-aware"
                            }
                        )
                    )
                else:
                    # normal paragraph
                    if len(p.strip()) > 30:
                        texts.append(
                            Document(
                                page_content=p.strip(),
                                metadata={
                                    "source": doc.metadata.get("source", "-"),
                                    "page": doc.metadata.get("page", "-"),
                                    "modality": "text"
                                }
                            )
                        )
    return texts
# 🔥 Tambahkan di sini
def hybrid_chunking(all_docs, max_table_rows=15):
    """
    Hybrid Chunking:
    - Text => dipecah per paragraf (>=30 karakter).
    - Table => dipotong per batch beberapa baris, tapi header selalu dibawa.
    - Image => tetap jadi marker.
    """
    texts = []
    for doc in all_docs:
        mod = doc.metadata.get("modality")

        if mod == "text":
            paras = [p.strip() for p in doc.page_content.split("\n\n") if len(p.strip()) > 30]
            for p in paras:
                texts.append(
                    Document(
                        page_content=p,
                        metadata={**doc.metadata, "modality": "text"}
                    )
                )

        elif mod == "table":
            rows = doc.page_content.split("\n")
            if not rows:
                continue

            header = rows[0]  # baris pertama sebagai header
            data_rows = rows[1:]

            if len(data_rows) > max_table_rows:
                for i in range(0, len(data_rows), max_table_rows):
                    chunk = "\n".join([header] + data_rows[i:i+max_table_rows])
                    if chunk.strip():
                        texts.append(
                            Document(
                                page_content=chunk,
                                metadata={**doc.metadata, "modality": "table"}
                            )
                        )
            else:
                texts.append(doc)

        else:  # image
            texts.append(doc)

    return texts
# ========== Vectorstore Pipeline ==========
def process_documents_and_index(uploaded_files, embedding_model: str, chunking_strategy: str):
    all_docs = []

    # Bersihkan memori & Chroma dir unik
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if os.path.exists(VECTORSTORE_DIR):
            shutil.rmtree(VECTORSTORE_DIR)
    except Exception as e:
        st.error(f"🚫 Gagal menghapus ChromaDB: {e}")
        return None

    # Simpan file dan ekstrak (JANGAN dihapus, agar bisa diunduh)
    for file in uploaded_files:
        filepath = os.path.join(UPLOAD_FOLDER, file.name)
        with open(filepath, "wb") as f:
            f.write(file.read())
        docs = extract_text_table_image(filepath)
        all_docs.extend(docs)

    if not all_docs:
        return None

    # Pilih strategi chunking
    if chunking_strategy == "Recursive":
        texts = recursive_chunking(all_docs, chunk_size=1000, chunk_overlap=200)
    elif chunking_strategy == "Table-Aware":
        texts = table_aware_chunking(all_docs)
    elif chunking_strategy == "Modality-Specific":
        texts = modality_specific_chunking(all_docs)
    elif chunking_strategy == "Hybrid":
        texts = hybrid_chunking(all_docs, max_table_rows=10)
    else:
        st.warning(f"Strategi chunking tidak dikenali: {chunking_strategy}. Menggunakan Recursive.")
        texts = recursive_chunking(all_docs, chunk_size=1000, chunk_overlap=200)


    st.session_state.source_docs = texts

    # Buat embeddings & vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    db = Chroma.from_documents(texts, embeddings, collection_name="rag-chroma", persist_directory=VECTORSTORE_DIR)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    # Bersihkan memori
    del all_docs, texts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return retriever

# ========== Prompt Template & LLM Caller ==========
def build_system_prompt(context: str, user_question: str) -> str:
    """
    1. Analisis Permintaan Pengguna:
    Identifikasi kata kunci utama dalam pertanyaan pengguna. Misalnya, "langkah-langkah pemulihan," "terhambatnya layanan," dan "Master Data Operations."
    Periksa apakah pertanyaan tersebut merujuk pada jenis bencana tertentu, seperti bencana alam, non-alam, atau sosial. Dalam kasus ini, layanan Master Data Operations terhambat karena bencana eksternal (bencana alam, darurat lingkungan, pandemi) atau internal (gangguan sistem/jaringan).
    Cari BCP (Business Continuity Plan) yang secara spesifik menanggapi proses bisnis tersebut.
    2. Ekstraksi Informasi dari Dokumen:
    Telusuri dokumen yang diberikan untuk menemukan bagian yang relevan. Gunakan daftar isi (Daftar Isi) dan pencarian teks untuk menemukan BCP yang terkait dengan "Layanan SS Master Data Operations."
    Setelah menemukan bagian yang benar, perhatikan judul "Recovery Procedure" atau "Langkah-langkah Pemulihan".
    Ekstrak informasi berikut secara terstruktur:
    Solusi Pemulihan: Identifikasi solusi yang diusulkan, yaitu "Melaksanakan Layanan SS MDO dengan Prosedur Request Melalui Email Kantor dan Pencatatan Request Manual".
    Lankah-langkah Pemulihan: Salin setiap langkah pemulihan secara berurutan sesuai nomornya (misalnya, Koordinasi awal, Penentuan Strategi, Penugasan PIC, dll.).
    Penanggung Jawab (PIC): Catat siapa yang bertanggung jawab untuk setiap langkah. Perhatikan nama-nama jabatan yang spesifik, seperti "Manager Master Data Operations" atau "Sr. Officer I SD Master Data".
    Waktu Pemulihan: Ambil durasi waktu yang ditetapkan untuk setiap langkah dan total waktu pemulihan (Recovery Time Objective).
    Dokumen Terkait: Cantumkan dokumen referensi yang disebutkan dalam setiap langkah.
    3. Penyusunan Respons:
    Jika hanya ada satu versi recovery procedure, sajikan langsung secara rinci.
    Jika ditemukan lebih dari satu versi:
    a. Identifikasi versi yang paling relevan atau paling mirip dengan pertanyaan pengguna
    (misalnya berdasarkan fungsi, layanan, atau kata kunci yang sama).
    b. Sajikan versi yang paling relevan tersebut secara rinci:
        langkah pemulihan berurutan
        PIC
        waktu
        dokumen terkait
    c. Jika ada versi lain yang berbeda, sebutkan secara singkat bahwa versi tersebut ada,
        dan jelaskan perbedaan utamanya dalam 2 hingga 3 kalimat (misalnya PIC berbeda, fungsi berbeda, RTO berbeda).
        Jangan uraikan seluruh detailnya jika tidak diminta.
    Selalu gunakan sitasi `` dengan nomor halaman dari dokumen untuk setiap informasi yang diambil langsung.
    Tutup dengan ringkasan singkat (solusi utama + konteks pemulihan)
    """
    return f"""

KONTEKS DOKUMEN:
{context}

PERTANYAAN PENGGUNA:
{user_question}
""".strip()

def format_sources_block(docs):
    # Gabungkan sumber unik (file, halaman)
    lines = []
    for d in docs:
        src = d.metadata.get("source", "-")
        page = d.metadata.get("page", "-")
        lines.append(f"- Halaman {page}, File: {src}")
    # Hilangkan duplikat sambil jaga urutan
    seen = set()
    unique_lines = []
    for ln in lines:
        if ln not in seen:
            seen.add(ln)
            unique_lines.append(ln)
    return "\n".join(unique_lines) if unique_lines else "- (Tidak ada sumber terdeteksi)"

def generate_response(user_prompt, docs, model_choice: str):
    context = "\n\n---\n\n".join([doc.page_content for doc in docs])
    system_prompt = build_system_prompt(context, user_prompt)
    sources_block = format_sources_block(docs)

    # Gemini (Cloud)
    if model_choice == "gemini-2.0-flash":
        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY belum diset. Tambahkan di environment variable atau st.secrets."
        try:
            model_obj = genai.GenerativeModel(
                "gemini-2.0-flash",
                generation_config={
                    "temperature": 0.01,
                    "top_p": 0.9,
                    "top_k": 40,
                }
            )
            resp = model_obj.generate_content(system_prompt)
            answer = (resp.text or "").strip()
        except Exception as e:
            return f"Error Gemini: {e}"

    # Ollama (Local): deepseek-r1:7b / tinyllama
    else:
        payload = {
            "model": model_choice,
            "prompt": user_prompt,
            "system": system_prompt,
            "stream": False,
            "temperature": 0.01,
            "top_k": 40,
            "top_p": 0.9
        }
        try:
            res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            res.raise_for_status()
            answer = res.json().get("response", "")
        except Exception as e:
            return f"Error Ollama: {e}"

    # Bersihkan & tambahkan Sumber + Penutup (dipastikan muncul)
    answer = clean_response(answer).rstrip()
    answer += f"\n\n📚 **Sumber:**\n{sources_block}\n\nApakah ada yang ingin Anda tanyakan kembali?"
    return answer

# ========== Sidebar ==========
with st.sidebar:
    try:
        st.image("logo_pertamina.png", width=250)
    except:
        st.warning("Logo tidak ditemukan.")
    st.markdown("## Chatbot SMKB BCMS")
    menu = st.radio("", ["Tanya Jawab", "Ruang Kerja", "Tentang"], label_visibility="collapsed")
    st.markdown(
        """
        <hr style="margin-top: 20px; margin-bottom: 10px;">
        <p style="font-size: 13px;">
            📩 <strong>Kontak:</strong><br>
            <a href="mailto:khairunnisaauliarhma@gmail.com">khairunnisaauliarhma@gmail.com</a>
        </p>
        """,
        unsafe_allow_html=True
    )

# ========== Session States ==========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    # default ke gemini jika API tersedia, jika tidak pakai deepseek
    st.session_state.selected_model = "gemini-2.0-flash" if GEMINI_API_KEY else "deepseek-r1:7b"
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

# ========== Halaman Tanya Jawab ==========
if menu == "Tanya Jawab":
    logo_path = "logo_pertamina_bulat.png"
    if os.path.exists(logo_path):
        logo_base64 = load_image_as_base64(logo_path)
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; gap: 12px; margin-bottom: 12px;">
                <img src="data:image/png;base64,{logo_base64}"
                    style="width: 40px; height: 40px; border-radius: 50%; object-fit: contain;" />
                <h2 style="margin: 0; line-height: 40px;">Knowledge Center SMKB</h2>
            </div>
            <p style="text-align: center; color: #666;">
                Pusat informasi terintegrasi berbasis AI untuk mendukung implementasi SMKB (ISO 22301) di lingkungan PT Pertamina (Persero).
            </p>
            """,
            unsafe_allow_html=True
        )
    else:
        st.title("📘 Knowledge Center SMKB")

    retriever = st.session_state.get("retriever", None)
    source_docs = st.session_state.get("source_docs", [])

    if not retriever or not source_docs:
        st.warning("⚠️ Silakan unggah dokumen terlebih dahulu di Ruang Kerja.")
        st.stop()

    # 👉 Tambahkan blok preview PDF di sini
    workspace_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf")]
    if workspace_files:
        selected_file = workspace_files[-1]  # otomatis ambil file terakhir
        file_path = os.path.join(UPLOAD_FOLDER, selected_file)

        doc_pdf = fitz.open(file_path)
        if "page_num" not in st.session_state:
            st.session_state.page_num = 0
        total_pages = len(doc_pdf)

    
        st.caption(f"Halaman {st.session_state.page_num+1} dari {total_pages}")

        page_obj = doc_pdf[st.session_state.page_num]
        pix = page_obj.get_pixmap(matrix=fitz.Matrix(1, 1))
        img_bytes = pix.tobytes("png")
        st.image(img_bytes, use_container_width=True)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Prev"):
                if st.session_state.page_num > 0:
                    st.session_state.page_num -= 1
                    st.rerun()

        with col2:
            page_input = st.number_input(
                " ",  # label kosong biar nggak makan tempat
                min_value=1,
                max_value=total_pages,
                value=st.session_state.page_num + 1,
                step=1,
                label_visibility="collapsed"  # sembunyikan label
            )
            # update halaman
            if page_input - 1 != st.session_state.page_num:
                st.session_state.page_num = page_input - 1
                st.rerun()

        with col3:
            if st.button("Next ➡️"):
                if st.session_state.page_num < total_pages - 1:
                    st.session_state.page_num += 1
                    st.rerun()

        doc_pdf.close()

    # 👉 Setelah preview baru tampilkan input pertanyaan
    query = st.text_input("Masukkan pertanyaan Anda", value=st.session_state.get("last_query", ""), key="user_query")
    submit = st.button("Tanyakan")

    if submit and query:
        start_retrieval = time.time()
        relevant_docs = retriever.get_relevant_documents(query)
        retrieval_time = time.time() - start_retrieval

        start_infer = time.time()
        answer = generate_response(query, relevant_docs, st.session_state.selected_model)
        infer_time = time.time() - start_infer

        # Simpan jawaban dan query ke session_state
        st.session_state.last_answer = answer
        st.session_state.last_query = query
        st.session_state.chat_history.append((query, answer))
        st.session_state.retrieval_time = retrieval_time
        st.session_state.infer_time = infer_time
        st.session_state.relevant_docs = relevant_docs

    # Ambil relevant_docs dari session_state jika tidak ada query baru
    relevant_docs = st.session_state.get("relevant_docs", [])

    if st.session_state.last_answer:
        st.markdown("### ✅ Jawaban:")
        st.markdown(st.session_state.last_answer)
        st.info(f"Retrieval time: {st.session_state.get('retrieval_time', 0):.2f} detik | Inference time: {st.session_state.get('infer_time', 0):.2f} detik")
    with st.expander("📚 Sumber Dokumen"):
        for doc in relevant_docs:
            file_name = doc.metadata.get("source", "-")
            page = doc.metadata.get("page", "-")
            st.markdown(f"- Halaman {page}, File: **{file_name}**")

        unique_files = {doc.metadata.get("source", "-") for doc in relevant_docs}
        for file_name in unique_files:
            file_path = os.path.join(UPLOAD_FOLDER, file_name)
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    st.download_button(
                        f"⬇️ Unduh Seluruh {file_name}",
                        data=f,
                        file_name=file_name,
                        mime="application/pdf",
                        key=f"download_full_{file_name}"
                    )

# ========== Halaman Ruang Kerja ==========
elif menu == "Ruang Kerja":
    password = st.text_input("🔐 Masukkan Password Akses", type="password")
    if password != "pertamina2024":
        st.warning("Password salah. Akses ditolak.")
        st.stop()

    st.markdown("## 🔧 Ruang Kerja")
    uploaded_files = st.file_uploader("📂 Unggah Dokumen PDF", type="pdf", accept_multiple_files=True)

    embedding_model = st.selectbox(
        "Pilih model embedding",
        [
            "intfloat/e5-base-v2",          # 1
            "sentence-transformers/all-MiniLM-L6-v2"  ,    
          "BAAI/bge-base-en-v1.5"  # 2
        ],
        index=0
    )

    chunking_strategy = st.selectbox(
        "Pilih metode chunking",
        ["Recursive", "Table-Aware", "Modality-Specific", "Hybrid"],
        index=3
    )

    model_choice = st.selectbox(
        "Model yang digunakan",
        ["gemini-2.0-flash", "deepseek-r1:7b", "tinyllama"],
        index=["gemini-2.0-flash", "deepseek-r1:7b", "tinyllama"].index(st.session_state.selected_model)
    )

    if uploaded_files:
        with st.spinner("📄 Memproses dokumen..."):
            retriever = process_documents_and_index(uploaded_files, embedding_model, chunking_strategy)
            if retriever:
                st.session_state.retriever = retriever
                st.session_state.selected_model = model_choice
                st.success("✅ Dokumen berhasil diproses dan siap digunakan.")
            else:
                st.error("❌ Gagal memproses dokumen.")

# ========== Halaman Ekspor Data ==========
elif menu == "Ekspor Data":
    password = st.text_input("🔐 Masukkan Password untuk Ekspor", type="password")
    if password != "export2024":   # ganti password sesuai kebutuhan
        st.warning("Password salah. Akses ditolak.")
        st.stop()

    st.markdown("## 📊 Ekspor Data Tanya Jawab")
    if st.session_state.chat_history:
        # Convert ke DataFrame
        df = pd.DataFrame([{"Pertanyaan": q, "Jawaban": a} for q, a in st.session_state.chat_history])

        # Simpan ke Excel (in-memory)
        from io import BytesIO
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="QnA")

        # Download button
        st.download_button(
            label="⬇️ Unduh Excel Tanya Jawab",
            data=output.getvalue(),
            file_name="chat_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_qna"
        )
    else:
        st.info("Belum ada data tanya jawab untuk diekspor.")

# ========== Halaman Tentang ==========
elif menu == "Tentang":
    st.markdown("""
    ## ℹ️ Tentang Aplikasi
    Chatbot SMKB BCMS adalah asisten cerdas untuk membantu interaksi dengan dokumen Sistem Manajemen Kelangsungan Bisnis (ISO 22301).

    **Fitur Utama:**
    - 🔐 Ruang kerja dengan login
    - 📄 Multi-upload PDF (dokumen disimpan lokal untuk diunduh kembali)
    - 🔎 RAG penuh (jawaban 100% dari dokumen)
    - 🧩 3 strategi chunking: Recursive, Table-Aware, Modality-Specific
    - 🧠 2 embedding terbaik: intfloat/e5-base-v2 & BAAI/bge-base-en-v1.5
    - 🤖 Multi-LLM: Gemini, DeepSeek & TinyLlama 
    - 📚 Sumber dokumen ditampilkan & dapat diunduh
    """)



