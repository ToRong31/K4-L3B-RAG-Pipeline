import os
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from src.task10_generation import generate_from_chunks, reorder_for_llm, format_context
from src.task9_retrieval_pipeline import retrieve
import src.task9_retrieval_pipeline as retrieval_pipeline


load_dotenv()

# Cấu hình trang giao diện
st.set_page_config(
    page_title="VinUni AI Advisor — RAG Pipeline",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS giao diện cao cấp, chuyên nghiệp
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Header Styling */
    .advisor-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #06b6d4 100%);
        padding: 24px 28px;
        border-radius: 16px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    .advisor-title {
        font-size: 26px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .advisor-subtitle {
        font-size: 14px;
        opacity: 0.9;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(8px);
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 12px;
        font-weight: 600;
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
    }

    /* Metric Card */
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-lbl {
        font-size: 12px;
        color: #64748b;
        margin-top: 2px;
    }

    /* Source Box */
    .source-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 12px;
    }
    .source-title {
        font-weight: 600;
        font-size: 14px;
        color: #1e3a8a;
    }
    .source-meta {
        font-size: 12px;
        color: #64748b;
        margin: 4px 0 8px 0;
    }
    .source-snippet {
        font-size: 13px;
        color: #334155;
        background: #ffffff;
        padding: 8px 12px;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        line-height: 1.5;
    }

    /* Pipeline Step Box */
    .pipeline-step {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    .step-header {
        font-size: 15px;
        font-weight: 600;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Khởi tạo session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_trace" not in st.session_state:
    st.session_state.last_trace = None
if "trace_history" not in st.session_state:
    st.session_state.trace_history = []

# SIDEBAR: Cấu hình thông số & Kiểm soát hệ thống
with st.sidebar:
    st.markdown("### ⚙️ Thông số RAG Engine")
    st.caption("Hiệu chỉnh các tham số Retrieval & Generation trong thời gian thực.")
    
    top_k = st.slider("🎯 Số lượng chunks (top_k)", min_value=3, max_value=10, value=5, help="Số lượng đoạn văn bản tối đa được đưa vào Context cho LLM.")
    score_threshold = st.slider("🛡️ Ngưỡng Fallback (Score Threshold)", min_value=0.1, max_value=0.9, value=0.3, step=0.05, help="Nếu điểm tương đồng Dense < ngưỡng này, hệ thống sẽ kích hoạt Fallback guardrail.")
    use_rerank = st.toggle("⚡ Bật hybrid retrieval", value=True, help="Bật BM25 + RRF; Cohere chạy sau RRF nếu đã cấu hình. Tắt để tìm kiếm dense-only.")
    
    st.divider()
    
    st.markdown("### 🧠 Mô hình & Nền tảng")
    provider = os.getenv("LLM_PROVIDER", "openai").upper()
    model = os.getenv("LLM_MODEL") or "gpt-4o-mini"
    dense_backend = os.getenv("DENSE_BACKEND", "shared").lower()
    embedding_model = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    e5_model = os.getenv("E5_MODEL", "intfloat/multilingual-e5-large")
    if dense_backend == "e5":
        embedding_model, e5_model = e5_model, embedding_model
    if os.getenv("MULTI_DENSE_ENABLED", "false").lower() == "true":
        embedding_model = f"{embedding_model} + {e5_model}"
    bm25_backend = "Elasticsearch BM25"
    if os.getenv("BM25_LOCAL_FALLBACK", "true").lower() == "true":
        bm25_backend += " (fallback local)"
    dense_weight = os.getenv("HYBRID_DENSE_WEIGHT", "0.7")
    bm25_weight = os.getenv("HYBRID_BM25_WEIGHT", "0.3")
    fusion_status = (
        f"Weighted RRF (k=60; dense {dense_weight}, BM25 {bm25_weight})"
        if use_rerank else "Tắt (dense-only)"
    )
    cohere_model = os.getenv("COHERE_RERANK_MODEL", "rerank-v4.0-fast")
    if not use_rerank:
        cohere_status = "Tắt (dense-only)"
    elif os.getenv("COHERE_RERANK_ENABLED", "false").lower() != "true":
        cohere_status = "Tắt trong .env"
    elif not os.getenv("COHERE_API_KEY"):
        cohere_status = "Tắt (chưa có API key)"
    else:
        cohere_status = f"Bật: {cohere_model} (sau RRF)"
    st.markdown(f"""
    - **LLM Engine:** `{provider}`
    - **Model:** `{model}`
    - **Embedding:** `{embedding_model}`
    - **Vector Store:** `ChromaDB (Cosine)`
    - **Lexical Search:** `{bm25_backend}`
    - **Fusion:** `{fusion_status}`
    - **Cohere Rerank:** `{cohere_status}`
    """)
    st.caption("PageIndex fallback trả kết quả trực tiếp; nhánh này không chạy Cohere.")
    
    st.divider()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Xóa chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.last_trace = None
            st.rerun()
    with col_btn2:
        if st.button("🔄 Làm mới", width="stretch"):
            st.rerun()

# HEADER BANNER CHÍNH
st.markdown("""
<div class="advisor-header">
    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
            <h1 class="advisor-title">🎓 Trợ Lý Tư Vấn Học Vụ & Chính Sách AI</h1>
            <p class="advisor-subtitle">Hệ thống hỏi đáp thông minh dựa trên bằng chứng minh bạch (Grounded RAG Pipeline with Citation & Fallback)</p>
        </div>
        <div class="status-badge">
            <span class="status-dot"></span> Pipeline Ready
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# TẠO 2 TABS CHÍNH: CHATBOT vs PIPELINE OBSERVABILITY
tab_chat, tab_telemetry = st.tabs(["💬 Trò chuyện tư vấn (Live Chat)", "🔬 Giám sát Pipeline & Logs (RAG Observability)"])

# ==========================================
# TAB 1: GIAO DIỆN CHATBOT TƯ VẤN THỰC TẾ
# ==========================================
with tab_chat:
    # Gợi ý câu hỏi mẫu khi chưa có tin nhắn
    if not st.session_state.messages:
        st.markdown("##### 💡 Gợi ý câu hỏi nhanh:")
        c1, c2, c3 = st.columns(3)
        sample_query = None
        with c1:
            if st.button("📜 Tiêu chuẩn xét học bổng là gì?", width="stretch"):
                sample_query = "Tiêu chuẩn xét học bổng là gì?"
        with c2:
            if st.button("💰 Quy định và thời hạn nộp học phí?", width="stretch"):
                sample_query = "Quy định và thời hạn nộp học phí?"
        with c3:
            if st.button("❓ Thời tiết hôm nay ở Hà Nội thế nào?", width="stretch", help="Câu hỏi ngoài domain để test Safe Refusal"):
                sample_query = "Thời tiết hôm nay ở Hà Nội thế nào?"
    else:
        sample_query = None

    # Hiển thị lịch sử tin nhắn
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍🎓" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])
            
            # Hiển thị trích dẫn nguồn
            if message["role"] == "assistant" and message.get("sources"):
                srcs = message["sources"]
                ret_src = message.get("retrieval_mode", message.get("retrieval_source", "hybrid"))
                with st.expander(f"📚 Xem {len(srcs)} tài liệu trích dẫn xác thực (Phương thức: {ret_src.upper()})"):
                    for idx, src in enumerate(srcs, 1):
                        meta = src.get("metadata", {})
                        title = meta.get("title", "Tài liệu quy định")
                        source = meta.get("source", "N/A")
                        url = meta.get("url")
                        score = src.get("score", 0.0)
                        method = src.get("retrieval_method", "N/A")
                        
                        st.markdown(f"""
                        <div class="source-box">
                            <div class="source-title">#{idx}. {title}</div>
                            <div class="source-meta">File: <b>{source}</b> | Score đầu ra: <b>{score:.4f}</b> | Kênh: <b>{method.upper()}</b> {f'| <a href="{url}" target="_blank">Mở link nguồn</a>' if url else ''}</div>
                            <div class="source-snippet">{src.get("content", "").strip()}</div>
                        </div>
                        """, unsafe_allow_html=True)

    # Ô nhập tin nhắn
    user_input = st.chat_input("Hỏi tôi về chính sách học bổng, học phí, quy chế sinh viên...")
    active_query = sample_query or user_input

    if active_query:
        # 1. Lưu và hiển thị câu hỏi người dùng
        st.session_state.messages.append({"role": "user", "content": active_query})
        with st.chat_message("user", avatar="🧑‍🎓"):
            st.markdown(active_query)

        # 2. Xử lý qua RAG Pipeline
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("🔍 Đang truy xuất tài liệu, chạy RRF Reranking và kiểm chứng trích dẫn..."):
                start_time = time.time()
                
                # Truy xuất theo đúng tuỳ chọn trên giao diện, rồi tạo câu trả lời từ các chunks đó.
                try:
                    chunks = retrieve(active_query, top_k=top_k, score_threshold=score_threshold, use_reranking=use_rerank)
                except Exception:
                    chunks = []
                retrieval_details = getattr(retrieval_pipeline, "get_last_retrieval_trace", lambda: {})()
                result = generate_from_chunks(active_query, chunks)
                latency = time.time() - start_time
                
                answer = result["answer"]
                sources = result.get("sources", [])
                retrieval_source = result.get("retrieval_source", "none")
                retrieval_mode = (
                    "pageindex" if retrieval_source == "pageindex"
                    else "dense-only" if not use_rerank and sources
                    else retrieval_source
                )
                
                # Lưu trace chi tiết cho tab telemetry
                trace_data = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "query": active_query,
                    **retrieval_details,
                    "answer": answer,
                    "top_k": top_k,
                    "score_threshold": score_threshold,
                    "use_reranking": use_rerank,
                    "latency_sec": round(latency, 3),
                    "sources_count": len(sources),
                    "retrieval_source": retrieval_source,
                    "retrieval_mode": retrieval_mode,
                    "sources": sources,
                    "is_refusal": (len(sources) == 0 or retrieval_source == "none"),
                }
                st.session_state.last_trace = trace_data
                st.session_state.trace_history.append(trace_data)
                
                # Hiển thị câu trả lời
                st.markdown(answer)
                
                # Hiển thị trích dẫn trực tiếp dưới câu trả lời
                if sources:
                    with st.expander(f"📚 Xem {len(sources)} tài liệu trích dẫn xác thực (Phương thức: {retrieval_mode.upper()})"):
                        for idx, src in enumerate(sources, 1):
                            meta = src.get("metadata", {})
                            title = meta.get("title", "Tài liệu quy định")
                            source = meta.get("source", "N/A")
                            url = meta.get("url")
                            score = src.get("score", 0.0)
                            method = src.get("retrieval_method", "N/A")
                            
                            st.markdown(f"""
                            <div class="source-box">
                                <div class="source-title">#{idx}. {title}</div>
                            <div class="source-meta">File: <b>{source}</b> | Score đầu ra: <b>{score:.4f}</b> | Kênh: <b>{method.upper()}</b> {f'| <a href="{url}" target="_blank">Mở link nguồn</a>' if url else ''}</div>
                                <div class="source-snippet">{src.get("content", "").strip()}</div>
                            </div>
                            """, unsafe_allow_html=True)
                elif trace_data["is_refusal"]:
                    st.caption("🛡️ *Kích hoạt Safe Refusal Guardrail: Không có tài liệu chứng minh trong cơ sở tri thức.*")

        # Lưu vào lịch sử tin nhắn
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_source": retrieval_source,
            "retrieval_mode": retrieval_mode,
        })
        st.rerun()

# ==========================================
# TAB 2: GIÁM SÁT CHI TIẾT CÁC CHỈ SỐ RAG (OBSERVABILITY)
# ==========================================
with tab_telemetry:
    st.markdown("### 📊 Bảng Điều Khiển Giám Sát RAG Pipeline (Live Telemetry & Invariants)")
    st.caption("Theo dõi chi tiết dữ liệu qua từng chặng: Chunking → Dense & BM25 → RRF Fusion → Fallback Threshold → Citation.")

    trace = st.session_state.last_trace

    if not trace:
        st.info("Chưa có lượt truy vấn nào được thực hiện. Hãy gửi một câu hỏi ở tab trò chuyện để xem các chỉ số đo lường thực tế!")
    else:
        # Băng chỉ số nhanh (KPI Metrics)
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{trace['latency_sec']}s</div>
                <div class="metric-lbl">⏱️ Tổng độ trễ (Latency)</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{trace['sources_count']}</div>
                <div class="metric-lbl">📑 Chunks được chọn</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {'#10b981' if trace['retrieval_source'] == 'hybrid' else '#f59e0b'};">{trace.get('retrieval_mode', trace['retrieval_source']).upper()}</div>
                <div class="metric-lbl">📡 Retrieval Method</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            max_score = max([s.get('score', 0.0) for s in trace['sources']]) if trace['sources'] else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{max_score:.4f}</div>
                <div class="metric-lbl">🏆 Best Output Score ({trace.get('score_type') or 'unknown'})</div>
            </div>
            """, unsafe_allow_html=True)
        with m5:
            refusal_status = "SAFE REFUSAL" if trace['is_refusal'] else "GROUNDED"
            refusal_color = "#ef4444" if trace['is_refusal'] else "#10b981"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val" style="color: {refusal_color};">{refusal_status}</div>
                <div class="metric-lbl">🛡️ Guardrail Status</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        cohere_status = trace.get("cohere_status", "unknown")
        st.caption(f"Cohere: {cohere_status} · Loại điểm hiển thị: {trace.get('score_type') or 'unknown'}. Khi Cohere timeout, hệ thống dùng điểm RRF khoảng 0.01; không so trực tiếp với cosine hoặc relevance score.")

        st.markdown("#### 🔎 Query formulation dùng cho retrieval")
        st.write(f"**Tiếng Việt / BM25:** {trace.get('query_vi') or 'Không có'}")
        st.write(f"**Tiếng Anh / dense:** {trace.get('query_en') or 'Không tạo được; dense dùng query tiếng Việt'}")
        st.caption("Dense tìm bằng các query trong `dense_queries`; BM25 dùng `bm25_query`. Các giá trị này lấy từ chính lượt retrieve ở trên.")

        # CHI TIẾT TỪNG GIAI ĐOẠN TRONG PIPELINE
        col_left, col_right = st.columns([1, 1])

        with col_left:
            # Stage 1: Chunking & Preprocessing Invariant
            st.markdown("""
            <div class="pipeline-step">
                <div class="step-header">🔹 Stage 1: Chunking & Indexing Invariants (Task 4)</div>
                <p style="font-size: 13px; color: #475569; margin: 0;">
                    • <b>Chunk Size:</b> 500 ký tự | <b>Overlap:</b> 50 ký tự<br>
                    • <b>Strategy:</b> RecursiveCharacterTextSplitter (bảo tồn ranh giới đoạn/câu)<br>
                    • <b>Vector Space:</b> Cosine Distance trong ChromaDB Collection
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Stage 2: Dual-Path Search (Dense + BM25)
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-header">🔹 Stage 2: Hai đường tìm kiếm song song (Task 5 & 6)</div>
                <p style="font-size: 13px; color: #475569; margin: 0;">
                    • <b>Dense Search:</b> Embedding query và tính <code>max(0, 1 - distance)</code><br>
                    • <b>Lexical Search:</b> {bm25_backend} trên cùng tập corpus<br>
                    • <b>Ứng viên mỗi nhánh:</b> <code>top_k * 2</code> để phục vụ tái xếp hạng
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Stage 3: Reranking & RRF Formula
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-header">🔹 Stage 3: Hợp nhất Reciprocal Rank Fusion (Task 7)</div>
                <p style="font-size: 13px; color: #475569; margin: 0;">
                    • <b>Trạng thái:</b> {'Đã bật hybrid RRF' if trace.get('use_reranking', True) else 'Dense-only; bỏ qua BM25, RRF và Cohere'}<br>
                    • <b>Công thức khi bật:</b> <code>RRF(d) = Σ [ weight_i / (60 + rank_i) ]</code><br>
                    • <b>Invariant:</b> RRF chỉ chạy duy nhất 1 lần, lọc trùng ID và xếp giảm dần<br>
                    • <b>Lưu ý:</b> Điểm RRF chỉ đại diện thứ hạng, không dùng để kích hoạt fallback
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            # Stage 4: Fallback Guardrail
            fallback_used = trace['retrieval_source'] == 'pageindex'
            threshold_used = trace.get('score_threshold', score_threshold)
            best_dense_score = trace.get('best_dense_score')
            best_dense_label = f"{best_dense_score:.4f}" if best_dense_score is not None else "N/A"
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-header">🔹 Stage 4: Kiểm soát ngưỡng Fallback (Task 8 & 9)</div>
                <p style="font-size: 13px; color: #475569; margin: 0;">
                    • <b>Ngưỡng đã dùng (Threshold):</b> <code>{threshold_used}</code><br>
                    • <b>Best Dense Cosine gốc:</b> <code>{best_dense_label}</code><br>
                    • <b>Kết quả:</b> <span style="color: {'#f59e0b' if fallback_used else '#10b981'}; font-weight: 600;">
                        {'Đã dùng PageIndex fallback' if fallback_used else 'Không dùng kết quả PageIndex'}
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Stage 5: Context Reordering (Lost-in-the-middle)
            st.markdown("""
            <div class="pipeline-step">
                <div class="step-header">🔹 Stage 5: Front-Back Reordering & Citation (Task 10)</div>
                <p style="font-size: 13px; color: #475569; margin: 0;">
                    • <b>Lost-in-the-middle Defense:</b> Chunk quan trọng nhất đặt ở vị trí #1 (Front) và #2 (Back)<br>
                    • <b>Citation Enforcement:</b> Prompt ép LLM chỉ trả lời khi có trích dẫn<br>
                    • <b>Safe Refusal:</b> Khi thiếu chứng cứ, trả về thông báo từ chối thay vì bịa thông tin
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # BẢNG DỮ LIỆU TÀI LIỆU TRÍCH XUẤT THỰC TẾ
        st.markdown("#### 📑 Bảng dữ liệu Chunks được nạp vào Prompt Context")
        if trace['sources']:
            source_rows = []
            for i, s in enumerate(trace['sources'], 1):
                meta = s.get('metadata', {})
                source_rows.append({
                    "Hạng": f"#{i}",
                    "Chunk ID": s.get("id", "N/A"),
                    "File Nguồn": meta.get("source", "N/A"),
                    "Tiêu Đề": meta.get("title", "N/A"),
                    "Score": f"{s.get('score', 0.0):.4f}",
                    "Kênh": s.get("retrieval_method", "N/A"),
                    "Trích đoạn": s.get("content", "")[:120] + "..."
                })
            st.dataframe(source_rows, width="stretch")
        else:
            st.warning("Không có chunks nào được trích xuất (Trường hợp câu hỏi ngoài domain hoặc cơ sở dữ liệu chưa có dữ liệu tương ứng).")

        # Raw JSON Payload Viewer
        with st.expander("🛠️ Xem Payload JSON chi tiết (Dành cho Giảng viên / Đánh giá viên)"):
            st.json(trace)
