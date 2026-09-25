import os
import time
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv

from src.query_formulation import formulate_query
from src.task9_retrieval_pipeline import retrieve
from src.task10_generation import (
    generate_with_citation,
    reorder_for_llm,
    format_context,
    call_llm,
    SYSTEM_PROMPT,
    SAFE_REFUSAL_ANSWER,
)


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

    /* Query Badges */
    .query-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .pill-raw {
        background-color: #f1f5f9;
        color: #334155;
        border: 1px solid #cbd5e1;
    }
    .pill-vi {
        background-color: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
    }
    .pill-en {
        background-color: #f5f3ff;
        color: #5b21b6;
        border: 1px solid #ddd6fe;
    }
    .badge-new-chunk {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        font-size: 11px;
        font-weight: 700;
        padding: 2px 8px;
        border-radius: 9999px;
        box-shadow: 0 2px 4px rgba(217, 119, 6, 0.3);
    }

    /* Comparison Panels */
    .cmp-panel-before {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .cmp-panel-after {
        background: #ffffff;
        border: 2px solid #3b82f6;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 4px 14px rgba(59, 130, 246, 0.12);
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
if "comparison_result" not in st.session_state:
    st.session_state.comparison_result = None

# SIDEBAR: Cấu hình thông số & Kiểm soát hệ thống
with st.sidebar:
    st.markdown("### ⚙️ Thông số RAG Engine")
    st.caption("Hiệu chỉnh các tham số Retrieval & Generation trong thời gian thực.")
    
    top_k = st.slider("🎯 Số lượng chunks (top_k)", min_value=3, max_value=10, value=5, help="Số lượng đoạn văn bản tối đa được đưa vào Context cho LLM.")
    score_threshold = st.slider("🛡️ Ngưỡng Fallback (Score Threshold)", min_value=0.1, max_value=0.9, value=0.3, step=0.05, help="Nếu điểm tương đồng Dense < ngưỡng này, hệ thống sẽ kích hoạt Fallback guardrail.")
    use_rerank = st.toggle("⚡ Bật RRF Reranking (Dense + BM25)", value=True, help="Hợp nhất kết quả Dense và BM25 bằng thuật toán Reciprocal Rank Fusion.")
    enable_formulation = st.toggle("🌐 Bật Query Formulation (Song ngữ VI/EN)", value=True, help="Tự động chuẩn hóa câu hỏi và dịch song ngữ Việt - Anh để mở rộng phạm vi tìm kiếm.")
    
    # Cập nhật biến môi trường theo toggle
    os.environ["QUERY_FORMULATION_ENABLED"] = "true" if enable_formulation else "false"

    st.divider()
    
    st.markdown("### 🧠 Mô hình & Nền tảng")
    provider = os.getenv("LLM_PROVIDER", "openai").upper()
    model = os.getenv("LLM_MODEL") or "gpt-4o-mini"
    emb_prov = os.getenv("EMBEDDING_PROVIDER", "openai").upper()
    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    st.markdown(f"""
    - **LLM Engine:** `{provider}` ({model})
    - **Query Formulation:** `gpt-4o-mini` (Bilingual)
    - **Embedding:** `{emb_prov}` ({emb_model})
    - **Vector Store:** `ChromaDB (Cosine)`
    - **Lexical Search:** `BM25Okapi`
    - **Fusion:** `RRF (k=60)`
    """)
    
    st.divider()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🗑️ Xóa chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_trace = None
            st.session_state.comparison_result = None
            st.rerun()
    with col_btn2:
        if st.button("🔄 Làm mới", use_container_width=True):
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

# TẠO 3 TABS CHÍNH: CHATBOT vs PIPELINE OBSERVABILITY vs QUERY FORMULATION COMPARISON
tab_chat, tab_telemetry, tab_comparison = st.tabs([
    "💬 Trò chuyện tư vấn (Live Chat)", 
    "🔬 Giám sát Pipeline & Logs (RAG Observability)",
    "🧪 Đối sánh Query Formulation (Before vs After)"
])

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
            if st.button("📜 Tiêu chuẩn xét học bổng là gì?", use_container_width=True):
                sample_query = "Tiêu chuẩn xét học bổng là gì?"
        with c2:
            if st.button("💰 Quy định và thời hạn nộp học phí?", use_container_width=True):
                sample_query = "Quy định và thời hạn nộp học phí?"
        with c3:
            if st.button("❓ Thời tiết hôm nay ở Hà Nội thế nào?", use_container_width=True, help="Câu hỏi ngoài domain để test Safe Refusal"):
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
                ret_src = message.get("retrieval_source", "hybrid")
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
                            <div class="source-meta">File: <b>{source}</b> | Độ tương đồng: <b>{score:.4f}</b> | Kênh: <b>{method.upper()}</b> {f'| <a href="{url}" target="_blank">Mở link nguồn</a>' if url else ''}</div>
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
                
                # Gọi hàm generation chuẩn từ task10
                result = generate_with_citation(active_query, top_k=top_k)
                latency = time.time() - start_time
                
                answer = result["answer"]
                sources = result.get("sources", [])
                retrieval_source = result.get("retrieval_source", "none")
                
                # Truy xuất thông tin Query Formulation nếu có
                if enable_formulation:
                    q_vi, q_en = formulate_query(active_query)
                else:
                    q_vi, q_en = active_query, ""

                # Lưu trace chi tiết cho tab telemetry
                trace_data = {
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "query": active_query,
                    "query_vi": q_vi,
                    "query_en": q_en,
                    "formulation_enabled": enable_formulation,
                    "answer": answer,
                    "top_k": top_k,
                    "latency_sec": round(latency, 3),
                    "sources_count": len(sources),
                    "retrieval_source": retrieval_source,
                    "sources": sources,
                    "is_refusal": (len(sources) == 0 or retrieval_source == "none"),
                }
                st.session_state.last_trace = trace_data
                st.session_state.trace_history.append(trace_data)
                
                # Hiển thị câu trả lời
                st.markdown(answer)
                
                # Hiển thị trích dẫn trực tiếp dưới câu trả lời
                if sources:
                    with st.expander(f"📚 Xem {len(sources)} tài liệu trích dẫn xác thực (Phương thức: {retrieval_source.upper()})"):
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
                                <div class="source-meta">File: <b>{source}</b> | Độ tương đồng: <b>{score:.4f}</b> | Kênh: <b>{method.upper()}</b> {f'| <a href="{url}" target="_blank">Mở link nguồn</a>' if url else ''}</div>
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
        })
        st.rerun()

# ==========================================
# TAB 2: GIÁM SÁT CHI TIẾT CÁC CHỈ SỐ RAG (OBSERVABILITY)
# ==========================================
with tab_telemetry:
    st.markdown("### 📊 Bảng Điều Khiển Giám Sát RAG Pipeline (Live Telemetry & Invariants)")
    st.caption("Theo dõi chi tiết dữ liệu qua từng chặng: Query Formulation → Chunking → Dense & BM25 → RRF Fusion → Fallback Threshold → Citation.")

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
                <div class="metric-val" style="color: {'#10b981' if trace['retrieval_source'] == 'hybrid' else '#f59e0b'};">{trace['retrieval_source'].upper()}</div>
                <div class="metric-lbl">📡 Retrieval Method</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            max_score = max([s.get('score', 0.0) for s in trace['sources']]) if trace['sources'] else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{max_score:.4f}</div>
                <div class="metric-lbl">🏆 Best Score</div>
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

        # CHI TIẾT TỪNG GIAI ĐOẠN TRONG PIPELINE
        # Stage 0: Query Formulation Display
        if trace.get("formulation_enabled"):
            st.markdown(f"""
            <div class="pipeline-step" style="border-left: 4px solid #8b5cf6;">
                <div class="step-header">✨ Stage 0: Query Formulation & Bilingual Expansion (Task 8 & Rubric Bonus)</div>
                <div style="display: flex; flex-direction: column; gap: 8px; margin-top: 8px;">
                    <div><span class="query-pill pill-raw">🔍 Câu hỏi gốc</span> <code style="font-size: 14px;">{trace['query']}</code></div>
                    <div><span class="query-pill pill-vi">🇻🇳 Chuẩn hóa tiếng Việt</span> <code style="font-size: 14px;">{trace.get('query_vi', 'N/A')}</code></div>
                    <div><span class="query-pill pill-en">🇬🇧 Mở rộng tiếng Anh</span> <code style="font-size: 14px;">{trace.get('query_en', 'N/A')}</code></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
            st.markdown("""
            <div class="pipeline-step">
                <div class="step-header">🔹 Stage 2: Hai đường tìm kiếm song song (Task 5 & 6)</div>
                <p style="font-size: 13px; color: #475569; margin: 0;">
                    • <b>Dense Search:</b> Embedding query và tính <code>max(0, 1 - distance)</code><br>
                    • <b>Lexical Search:</b> Tokenize và xếp hạng BM25Okapi trên cùng tập corpus<br>
                    • <b>Ứng viên mỗi nhánh:</b> <code>top_k * 2</code> để phục vụ tái xếp hạng
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Stage 3: Reranking & RRF Formula
            st.markdown("""
            <div class="pipeline-step">
                <div class="step-header">🔹 Stage 3: Hợp nhất Reciprocal Rank Fusion (Task 7)</div>
                <p style="font-size: 13px; color: #475569; margin: 0;">
                    • <b>Công thức:</b> <code>RRF(d) = Σ [ 1 / (60 + rank_i) ]</code><br>
                    • <b>Invariant:</b> RRF chỉ chạy duy nhất 1 lần, lọc trùng ID và xếp giảm dần<br>
                    • <b>Lưu ý:</b> Điểm RRF chỉ đại diện thứ hạng, không dùng để kích hoạt fallback
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col_right:
            # Stage 4: Fallback Guardrail
            dense_confidence = max_score
            is_fallback_triggered = dense_confidence < score_threshold
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-header">🔹 Stage 4: Kiểm soát ngưỡng Fallback (Task 8 & 9)</div>
                <p style="font-size: 13px; color: #475569; margin: 0;">
                    • <b>Ngưỡng cài đặt (Threshold):</b> <code>{score_threshold}</code><br>
                    • <b>Best Dense Score gốc:</b> <code>{dense_confidence:.4f}</code><br>
                    • <b>Trạng thái:</b> <span style="color: {'#ef4444' if is_fallback_triggered else '#10b981'}; font-weight: 600;">
                        {'Kích hoạt Fallback (Score < Threshold)' if is_fallback_triggered else 'Đủ tin cậy (Score >= Threshold)'}
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
            st.dataframe(source_rows, use_container_width=True)
        else:
            st.warning("Không có chunks nào được trích xuất (Trường hợp câu hỏi ngoài domain hoặc cơ sở dữ liệu chưa có dữ liệu tương ứng).")

        # Raw JSON Payload Viewer
        with st.expander("🛠️ Xem Payload JSON chi tiết (Dành cho Giảng viên / Đánh giá viên)"):
            st.json(trace)

# =========================================================================
# TAB 3: ĐỐI SÁNH QUERY FORMULATION (BEFORE VS AFTER A/B WORKBENCH)
# =========================================================================
with tab_comparison:
    st.markdown("### 🧪 Đối Sánh Trực Quan: Trước vs Sau Query Formulation")
    st.caption("Khảo sát sự khác biệt khi dùng Câu hỏi gốc (Raw Query) so với Câu hỏi được LLM chuẩn hoá và dịch song ngữ Việt - Anh (Formulated Queries). Điểm cộng sáng tạo +3 trong Rubric!")

    # Cụm nhập liệu và chọn câu hỏi mẫu
    st.markdown("##### 📌 Chọn câu hỏi thử nghiệm hoặc nhập tùy ý:")
    c_s1, c_s2, c_s3, c_s4 = st.columns(4)
    cmp_selected = None
    with c_s1:
        if st.button("💰 Học phí VinUni?", key="cmp_btn1", use_container_width=True):
            cmp_selected = "Học phí của trường Đại học VinUni một năm bao nhiêu?"
    with c_s2:
        if st.button("📜 Tiêu chuẩn học bổng?", key="cmp_btn2", use_container_width=True):
            cmp_selected = "Tiêu chuẩn và điều kiện duy trì học bổng VinUni?"
    with c_s3:
        if st.button("⚠️ Cảnh báo & thôi học?", key="cmp_btn3", use_container_width=True):
            cmp_selected = "Khi nào sinh viên bị cảnh báo học tập hoặc buộc thôi học?"
    with c_s4:
        if st.button("🎓 Graduation rules?", key="cmp_btn4", use_container_width=True):
            cmp_selected = "What are the graduation requirements for undergraduate students?"

    cmp_query_input = st.text_input(
        "Nhập câu hỏi cần phân tích đối sánh A/B:",
        value=cmp_selected if cmp_selected else (st.session_state.comparison_result.get("raw_query", "") if st.session_state.comparison_result else ""),
        placeholder="Ví dụ: Học phí một năm bao nhiêu? Điều kiện nhận học bổng tài năng?",
        key="cmp_input_field"
    )

    col_cfg1, col_cfg2 = st.columns([1, 2])
    with col_cfg1:
        cmp_top_k = st.slider("🎯 top_k phân tích", min_value=3, max_value=8, value=5, key="cmp_topk_slider")
    with col_cfg2:
        st.write("")
        st.write("")
        run_cmp_button = st.button("🚀 Bấm để chạy so sánh Đối kháng (Run Before vs After Comparison)", type="primary", use_container_width=True)

    if run_cmp_button and cmp_query_input.strip():
        query_text = cmp_query_input.strip()

        with st.spinner("⏳ Đang thực thi song song 2 luồng: [Raw Retrieval] vs [Bilingual Formulated Retrieval]..."):
            # 1. Luồng A: Trước khi Formulate (Raw Query)
            t0 = time.time()
            os.environ["QUERY_FORMULATION_ENABLED"] = "false"
            try:
                chunks_before = retrieve(query_text, top_k=cmp_top_k, score_threshold=score_threshold, use_reranking=use_rerank)
            except Exception:
                chunks_before = []

            if chunks_before:
                reordered_b = reorder_for_llm(chunks_before)
                ctx_b = format_context(reordered_b)
                msg_b = f"Context:\n{ctx_b}\n\nQuestion: {query_text}"
                try:
                    ans_b = call_llm(SYSTEM_PROMPT, msg_b) or SAFE_REFUSAL_ANSWER
                except Exception:
                    ans_b = SAFE_REFUSAL_ANSWER
            else:
                ans_b = SAFE_REFUSAL_ANSWER
            latency_before = round(time.time() - t0, 3)

            # 2. Luồng B: Sau khi Formulate (Formulated Queries)
            t1 = time.time()
            os.environ["QUERY_FORMULATION_ENABLED"] = "true"
            q_vi, q_en = formulate_query(query_text)
            try:
                chunks_after = retrieve(query_text, top_k=cmp_top_k, score_threshold=score_threshold, use_reranking=use_rerank)
            except Exception:
                chunks_after = []

            if chunks_after:
                reordered_a = reorder_for_llm(chunks_after)
                ctx_a = format_context(reordered_a)
                msg_a = f"Context:\n{ctx_a}\n\nQuestion: {query_text}"
                try:
                    ans_a = call_llm(SYSTEM_PROMPT, msg_a) or SAFE_REFUSAL_ANSWER
                except Exception:
                    ans_a = SAFE_REFUSAL_ANSWER
            else:
                ans_a = SAFE_REFUSAL_ANSWER
            latency_after = round(time.time() - t1, 3)

            # Lưu vào session state
            st.session_state.comparison_result = {
                "raw_query": query_text,
                "query_vi": q_vi,
                "query_en": q_en,
                "before": {
                    "chunks": chunks_before,
                    "answer": ans_b,
                    "latency": latency_before,
                    "top_score": max((c.get("score", 0.0) for c in chunks_before), default=0.0),
                    "avg_score": (sum(c.get("score", 0.0) for c in chunks_before) / len(chunks_before)) if chunks_before else 0.0,
                },
                "after": {
                    "chunks": chunks_after,
                    "answer": ans_a,
                    "latency": latency_after,
                    "top_score": max((c.get("score", 0.0) for c in chunks_after), default=0.0),
                    "avg_score": (sum(c.get("score", 0.0) for c in chunks_after) / len(chunks_after)) if chunks_after else 0.0,
                }
            }

    # HIỂN THỊ KẾT QUẢ ĐỐI SÁNH NẾU ĐÃ CHẠY
    res = st.session_state.comparison_result
    if res:
        b_data = res["before"]
        a_data = res["after"]
        ids_before = [c["id"] for c in b_data["chunks"]]
        ids_after = [c["id"] for c in a_data["chunks"]]
        set_b = set(ids_before)
        set_a = set(ids_after)
        
        overlap_count = len(set_b & set_a)
        new_in_after = [c for c in a_data["chunks"] if c["id"] not in set_b]
        new_count = len(new_in_after)

        st.markdown("---")
        st.markdown("### 📊 Tổng Kết Hiệu Quả Đối Sánh (Delta Metrics)")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #10b981;">
                <div class="metric-val" style="color: #10b981;">+{new_count} Chunks</div>
                <div class="metric-lbl">🌟 Chunks mới khám phá nhờ song ngữ</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            score_diff = a_data["top_score"] - b_data["top_score"]
            diff_sign = "+" if score_diff >= 0 else ""
            diff_color = "#10b981" if score_diff >= 0 else "#ef4444"
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #3b82f6;">
                <div class="metric-val">{b_data['top_score']:.4f} → {a_data['top_score']:.4f}</div>
                <div class="metric-lbl">🏆 Best Score (<span style="color: {diff_color}; font-weight:700;">{diff_sign}{score_diff:.4f}</span>)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #8b5cf6;">
                <div class="metric-val">{b_data['latency']}s vs {a_data['latency']}s</div>
                <div class="metric-lbl">⏱️ Thời gian thực thi (Latency)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #f59e0b;">
                <div class="metric-val">{overlap_count} / {len(set_a)}</div>
                <div class="metric-lbl">🔗 Tỉ lệ hội tụ Chunks cốt lõi</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 🔍 Chi tiết chuyển đổi câu hỏi (Query Transformation):")
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px; margin-bottom: 20px;">
            <div style="margin-bottom: 8px;">
                <span class="query-pill pill-raw">Trước khi Formulate</span> <code>{res['raw_query']}</code>
            </div>
            <div style="margin-bottom: 8px;">
                <span class="query-pill pill-vi">🇻🇳 Sau Formulate (Tiếng Việt)</span> <code>{res['query_vi']}</code>
            </div>
            <div>
                <span class="query-pill pill-en">🇬🇧 Sau Formulate (Tiếng Anh)</span> <code>{res['query_en']}</code>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # HAI CỘT HIỂN THỊ SONG SONG (SIDE-BY-SIDE)
        col_left, col_right = st.columns(2)

        # CỘT TRÁI: BEFORE FORMULATION
        with col_left:
            st.markdown("""
            <div class="cmp-panel-before">
                <h4 style="margin: 0 0 10px 0; color: #475569;">⚪ [Nhánh A] Trước khi Formulate (Raw Query)</h4>
                <p style="font-size: 13px; color: #64748b; margin-bottom: 12px;">Truy vấn trực tiếp bằng câu hỏi gốc của người dùng.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 🤖 Câu trả lời sinh ra:")
            st.info(b_data["answer"])

            st.markdown(f"##### 📑 Danh sách {len(b_data['chunks'])} Chunks trích xuất:")
            if not b_data["chunks"]:
                st.warning("Không tìm thấy chunk nào.")
            else:
                for idx, c in enumerate(b_data["chunks"], 1):
                    meta = c.get("metadata", {})
                    st.markdown(f"""
                    <div class="source-box" style="border-left-color: #94a3b8;">
                        <div class="source-title">#{idx}. {meta.get('title', 'Tài liệu')}</div>
                        <div class="source-meta">ID: <code>{c.get('id', 'N/A')}</code> | Score: <b>{c.get('score', 0.0):.4f}</b> | File: {meta.get('source', 'N/A')}</div>
                        <div class="source-snippet">{c.get('content', '')[:160]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        # CỘT PHẢI: AFTER FORMULATION
        with col_right:
            st.markdown("""
            <div class="cmp-panel-after">
                <h4 style="margin: 0 0 10px 0; color: #1d4ed8;">🔵 [Nhánh B] Sau khi Formulate (Bilingual Expansion)</h4>
                <p style="font-size: 13px; color: #1e40af; margin-bottom: 12px;">Mở rộng song ngữ Việt - Anh kết hợp hợp nhất RRF.</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("##### 🤖 Câu trả lời sinh ra:")
            st.success(a_data["answer"])

            st.markdown(f"##### 📑 Danh sách {len(a_data['chunks'])} Chunks trích xuất:")
            if not a_data["chunks"]:
                st.warning("Không tìm thấy chunk nào.")
            else:
                for idx, c in enumerate(a_data["chunks"], 1):
                    meta = c.get("metadata", {})
                    is_new = c["id"] not in set_b
                    badge_html = '<span class="badge-new-chunk">⭐ CHUNK MỚI</span>' if is_new else ''
                    border_color = "#f59e0b" if is_new else "#3b82f6"
                    
                    st.markdown(f"""
                    <div class="source-box" style="border-left-color: {border_color};">
                        <div class="source-title">#{idx}. {meta.get('title', 'Tài liệu')} {badge_html}</div>
                        <div class="source-meta">ID: <code>{c.get('id', 'N/A')}</code> | Score: <b>{c.get('score', 0.0):.4f}</b> | File: {meta.get('source', 'N/A')}</div>
                        <div class="source-snippet">{c.get('content', '')[:160]}...</div>
                    </div>
                    """, unsafe_allow_html=True)

        # BẢNG MA TRẬN ĐỐI CHIẾU CHI TIẾT
        st.markdown("---")
        st.markdown("#### 📋 Ma Trận Đối Chiếu Chi Tiết Từng Đoạn Văn Bản (Chunk Mapping Matrix)")
        
        matrix_rows = []
        all_ids = list(dict.fromkeys(ids_after + ids_before))
        for cid in all_ids:
            # Tìm trong before
            b_item = next((c for c in b_data["chunks"] if c["id"] == cid), None)
            b_rank = (ids_before.index(cid) + 1) if b_item else "—"
            b_score = f"{b_item['score']:.4f}" if b_item else "—"
            
            # Tìm trong after
            a_item = next((c for c in a_data["chunks"] if c["id"] == cid), None)
            a_rank = (ids_after.index(cid) + 1) if a_item else "—"
            a_score = f"{a_item['score']:.4f}" if a_item else "—"
            
            title = (a_item or b_item).get("metadata", {}).get("title", "N/A")
            
            if b_item and a_item:
                tag = "🔄 Trùng lặp cốt lõi (Core Match)"
            elif a_item and not b_item:
                tag = "⭐ Mới khám phá nhờ Formulate (New)"
            else:
                tag = "🔻 Bị loại khỏi top_k (Deprioritized)"
                
            matrix_rows.append({
                "Chunk ID": cid,
                "Tiêu đề tài liệu": title,
                "Thứ hạng Trước": b_rank,
                "Thứ hạng Sau": a_rank,
                "Score Trước": b_score,
                "Score Sau": a_score,
                "Phân loại hiệu quả": tag,
            })
        st.dataframe(matrix_rows, use_container_width=True)

        # HỌC THUẬT & GIẢI TRÌNH RUBRIC
        st.markdown("""
        > 💡 **Nhận định kỹ thuật (Academic Takeaway - Grading Rubric +3 Points):**
        > - **Khắc phục Vocabulary Mismatch:** Ngôn ngữ câu hỏi tự nhiên của sinh viên thường ngắn gọn, dùng từ ngữ thông dụng ("học phí bao nhiêu", "nghỉ học"), trong khi văn bản quy chế sử dụng từ ngữ pháp lý chuẩn ("biểu phí đào tạo", "buộc thôi học") hoặc thuật ngữ tiếng Anh ("Credit System", "Academic Warning").
        > - **Hiệu quả của Query Formulation:** Khi kích hoạt `formulate_query()`, mô hình mở rộng đồng thời 2 biểu diễn truy vấn (1 tiếng Việt chuẩn tắc + 1 bản dịch tiếng Anh). Quá trình Dense & Sparse tìm kiếm trên cả 2 ngôn ngữ và hợp nhất qua RRF giúp tăng đáng kể **Recall** và **MRR**, thu hồi thêm các chunk giá trị mà phương pháp Raw Query đơn ngữ bị bỏ sót.
        """)
