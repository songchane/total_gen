import streamlit as st
from book_writer import run_analysis

# PDF 생성 라이브러리
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from io import BytesIO


pdfmetrics.registerFont(TTFont("Nanum", "fonts/NanumGothic.ttf"))

def markdown_to_pdf(md_text: str) -> bytes:
    buffer = BytesIO()

    # 👉 Markdown의 Bold(**) 를 HTML <b></b> 태그로 변환
    import re
    md_text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", md_text)

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
    )

    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        "Korean",
        parent=styles["Normal"],
        fontName="Nanum",
        fontSize=12,
        leading=18,
        alignment=TA_LEFT,
    )

    title_style = ParagraphStyle(
        "Title",
        parent=styles["Heading1"],
        fontName="Nanum",
        fontSize=18,
        leading=22,
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Heading2"],
        fontName="Nanum",
        fontSize=16,
        leading=20,
    )

    items = []

    for line in md_text.split("\n"):
        line = line.strip()

        # 제목
        if line.startswith("# "):
            items.append(Paragraph(line[2:], title_style))
            items.append(Spacer(1, 10))

        elif line.startswith("## "):
            items.append(Paragraph(line[3:], subtitle_style))
            items.append(Spacer(1, 8))

        elif line.startswith("### "):
            # 이미 Bold 처리됨
            items.append(Paragraph(line[4:], body_style))
            items.append(Spacer(1, 6))

        # 리스트 항목
        elif line.startswith("- "):
            bullet_text = "• " + line[2:]
            items.append(Paragraph(bullet_text, body_style))
            items.append(Spacer(1, 4))

        elif line == "":
            items.append(Spacer(1, 10))

        else:
            items.append(Paragraph(line, body_style))
            items.append(Spacer(1, 6))

    doc.build(items)
    buffer.seek(0)
    return buffer.getvalue()

st.markdown(
    """
    <style>

        .main { background-color: #F2F4F7 !important; }

        .center-title {
            text-align: center;
            font-size: 2.3rem !important;
            font-weight: 800 !important;
            color: #2F3A4A !important;
            margin-top: -10px !important;
            margin-bottom: 10px !important;
        }

        div[data-baseweb="input"] > div {
            border-radius: 12px !important;
            border: 1.5px solid #CED4DA !important;
            background-color: #FAFAFA !important;
            padding: 6px !important;
        }

        .stButton>button {
            background: linear-gradient(135deg, #4A7DFF 0%, #647BFF 100%) !important;
            color: white !important;
            padding: 0.65rem 1.3rem !important;
            border-radius: 12px !important;
            border: none !important;
            font-weight: 600 !important;
            font-size: 1rem !important;
            margin-top: 8px !important;
        }

        .stButton>button:hover {
            background: linear-gradient(135deg, #3D6EE8 0%, #5169E9 100%) !important;
        }

        .report-box {
            padding: 2rem !important;
            background: #FFFFFFAA;
            backdrop-filter: blur(12px);
            border-radius: 16px !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.07) !important;
            border: 1px solid #E2E6EA !important;
            line-height: 1.6;
        }

    </style>
    """,
    unsafe_allow_html=True
)


st.markdown("<h1 class='center-title'>📊 상권 BI 트렌드 분석 보고서 생성기</h1>", unsafe_allow_html=True)


st.markdown("### 📝 분석 요청을 입력하세요")

user_input = st.text_input(
    "",
    placeholder="예: 성수동 카페 상권의 최근 1년 트렌드 분석 보고서를 만들어줘",
    label_visibility="collapsed",
)

submitted = st.button("🚀 보고서 생성하기")
st.markdown("---")


if submitted:
    if not user_input.strip():
        st.warning("⚠️ 분석 요청을 입력하세요!")
    else:
        with st.spinner("📡 상권 BI 분석 중입니다... 잠시만 기다려주세요 ⏳"):
            report = run_analysis(user_input)

        # 🔥 보고서를 세션에 저장
        st.session_state["report"] = report


if "report" in st.session_state:
    report = st.session_state["report"]

    st.markdown("### 📄 생성된 보고서")
    st.markdown(f"<div class='report-box'>{report}</div>", unsafe_allow_html=True)

    # 버튼과 보고서 사이 간격
    st.markdown("<div style='height:25px'></div>", unsafe_allow_html=True)

    # PDF 다운로드 버튼
    pdf_bytes = markdown_to_pdf(report)
    st.download_button(
        "📄 PDF 다운로드",
        data=pdf_bytes,
        file_name="market_trend_report.pdf",
        mime="application/pdf",
    )
