# book_writer.py — BI 상권 트렌드 분석 엔진


from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AnyMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
)
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from dotenv import load_dotenv

from tools import retrieve, web_search, add_web_pages_json_to_chroma

load_dotenv()


# 모델 & 상태 정의

llm = ChatOpenAI(model="gpt-4o")


class State(TypedDict):
    messages: List[AnyMessage]
    references: Dict[str, Any]  # {"queries": [...], "docs": [...]}
    user_request: Dict[str, Any]  # {"region": ..., "industry": ...}
    report: Optional[str]


# 1. 쿼리 파서 (Business Analyst)

def parse_query(state: State) -> State:
    print("\n\n============ PARSE QUERY (Business Analyst) ============")

    messages = state["messages"]
    user_last = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_last = m.content
            break

    prompt = PromptTemplate.from_template(
        """
        너는 '정성 기반 상권 BI 분석 챗봇'의 쿼리 파서이다.

        사용자의 질문을 기반으로 아래 JSON 형식만 정확히 출력하라.

        ```json
        {{
          "region": "예: 성수동, 용산구, 홍대입구, 부산 서면",
          "industry": "예: 카페, F&B, 외식업, 패션, 전체",
          "period": "예: 최근 6개월, 최근 1년, 2024년, 2023년 1~6월",
          "keywords": ["트렌드", "핫플", "소비자 패턴", "업종 변화"]
        }}
        ```

        - region은 상권 분석의 중심이 되는 주요 지역명으로 채워라.
        - industry는 질문에 명시된 업종(카페, 음식점 등)이 있으면 그대로, 없으면 '전체'로 채워라.
        - period는 질문에 '최근 1년', '최근 6개월' 등이 있으면 그대로, 없으면 '최근 1년'으로 채워라.
        - keywords는 최소 ["트렌드"] 한 개는 반드시 포함하라.

        사용자 질문: {user_last_comment}
        """
    )

    chain = prompt | llm | StrOutputParser()
    raw_json = chain.invoke({"user_last_comment": user_last})

    # ```json ... ``` 제거
    cleaned = raw_json.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        parsed = {
            "region": user_last,
            "industry": "전체",
            "period": "최근 1년",
            "keywords": ["트렌드"],
        }

    print(f"[Parsed Query] {parsed}")

    state["user_request"] = parsed
    state["messages"].append(AIMessage(content=f"[Parsed] {parsed}"))
    return state


# 2. Web Search (DuckDuckGo + JSON + Chroma)

def web_search_node(state: State) -> State:
    print("\n\n============ WEB SEARCH ============")

    req = state["user_request"]
    region = req.get("region", "")
    industry = req.get("industry", "")
    period = req.get("period", "최근 1년")
    keywords = req.get("keywords", ["트렌드"])

    # LLM 없이 Python에서 안전하게 검색어 구성
    base = f"{region} {industry}".strip()
    if not base:
        base = region or industry or "상권"

    queries = [
        f"{base} 상권 트렌드 {period}",
        f"{base} 카페 트렌드 {period}" if industry == "카페" else f"{base} 소비자 패턴 {period}",
        f"{base} 핫플 상권 변화 {period}",
    ]

    refs = state["references"]

    for q in queries:
        q = q.strip()
        if not q:
            continue
        print(f"[WebSearch] query={q}")
        try:
            # tools.web_search: (results, json_path) 가정
            _, json_path = web_search.invoke({"query": q})
            add_web_pages_json_to_chroma(json_path)
            refs["queries"].append(q)
        except Exception as e:
            print(f"[WebSearch Error] {e}")

    return state


# 3. Vector Search (RAG)

def vector_search_node(state: State) -> State:
    print("\n\n============ VECTOR SEARCH (RAG) ============")

    req = state["user_request"]
    region = req.get("region", "")
    industry = req.get("industry", "")
    period = req.get("period", "최근 1년")

    refs = state["references"]

    rag_queries = [
        f"{region} {industry} 상권 트렌드 {period}",
        f"{region} {industry} 소비자 행동 패턴 {period}",
        f"{region} {industry} 인기 업종 변화 {period}",
    ]

    for q in rag_queries:
        q = q.strip()
        if not q:
            continue
        print(f"[RAG] query={q}")
        try:
            docs = retrieve.invoke({"query": q, "top_k": 5})
            refs["queries"].append(q)
            refs["docs"].extend(docs)
        except Exception as e:
            print(f"[RAG Error] {e}")

    # 중복 제거
    seen = set()
    unique_docs = []
    for d in refs["docs"]:
        content = getattr(d, "page_content", "")
        if content not in seen:
            seen.add(content)
            unique_docs.append(d)
    refs["docs"] = unique_docs

    print(f"[RAG] 총 문서 수: {len(refs['docs'])}")
    return state


# 4. Content Strategist (BI 보고서 생성)

def content_strategist_node(state: State) -> State:
    print("\n\n============ CONTENT STRATEGIST ============")

    req = state["user_request"]
    refs = state["references"]

    region = req.get("region", "해당 지역")
    industry = req.get("industry", "전체")
    period = req.get("period", "최근 1년")

    # RAG 문서 요약 일부 추출
    doc_snippets = []
    for d in refs.get("docs", [])[:10]:
        text = getattr(d, "page_content", "")
        snippet = text[:500].replace("\n", " ")
        if snippet:
            doc_snippets.append(snippet)
    docs_text = "\n\n---\n\n".join(doc_snippets) if doc_snippets else "관련 문서가 거의 없음"

    prompt = PromptTemplate.from_template(
        """
        너는 '상권 BI 트렌드 분석' 전문 애널리스트다.

        아래 region, industry, period, 그리고 참고 문서(RAG)를 바탕으로
        실제 비즈니스 의사결정에 도움이 되는 인사이트 중심의 보고서를 작성하라.

        반드시 아래 목차를 따르되, 내용은 구체적이고 전략적 인사이트 위주로 채워라.

        # {region} 상권 트렌드 분석 보고서

        ## 1. 개요
        - 분석 지역: {region}
        - 업종 범위: {industry}
        - 분석 기간: {period}
        - 데이터 출처: DuckDuckGo Web RAG + GPT 분석
        - 분석 목적: 상권 트렌드 파악 및 유망 업종 전략 수립 지원

        ## 2. 상권 핵심 요약(Key Summary)
        - 해당 상권의 핵심 트렌드 Top 3~5
        - 소비자 행동 및 방문 패턴 요약
        - 인기 업종/콘셉트 요약
        - 상권의 강점과 약점 한 줄씩 정리

        ## 3. 환경 변화 분석(Macro & Local Trend)
        ### 3.1 외부 환경 및 개발 이슈
        - 교통 인프라, 재개발/도시계획, 상권 확장/축소 관련 이슈
        - 정책, 규제, 상업지역 조정 등 상권에 영향을 미치는 요소

        ### 3.2 소비자 행동 트렌드
        - 주 고객층(연령/직업/라이프스타일)의 특징
        - 소비 성향(가심비/가성비/SNS 인증/경험 중심 등)
        - 요일/시간대별 방문 패턴 변화

        ## 4. 상권 구조 및 경쟁 분석
        ### 4.1 핵심 상권 구역 2~4곳 분석
        - 각 구역의 상권 성격(관광/로컬/오피스/주거 등)
        - 업종 구성 비율과 특징
        - 신규 브랜드/팝업/폐점 등 최근 변화

        ### 4.2 경쟁도·포화도 분석
        - 업종별 경쟁 강도 (카페/디저트/F&B/패션 등)
        - 과밀/과소 구역에 대한 정성적 판단
        - 진입 장벽 및 차별화 포인트

        ## 5. 업종별 트렌드 세부 분석
        ### 5.1 카페 & 디저트
        - 인기 메뉴, 인테리어/콘셉트, 체류 시간 등
        - 소비자 니즈와 방문 동기

        ### 5.2 레스토랑 & F&B (해당 시)
        - 주목받는 음식 카테고리
        - 데이트/모임/일상 소비 등 방문 목적

        ### 5.3 패션/라이프스타일 (해당 시)
        - 로컬 브랜드, 팝업 스토어, 라이프스타일 변화

        ## 6. 기회 요인(Strength / Opportunity)
        - 성장 가능성이 높은 업종/콘셉트
        - 빠르게 증가하는 소비자 유형
        - 경쟁이 상대적으로 덜한 니치 영역

        ## 7. 리스크 요인(Weakness / Threat)
        - 경쟁 과열 업종
        - 유행 피로도 및 단기 트렌드 위험
        - 비용·임대료·입지 관련 리스크

        ## 8. GPT 추천 업종 Top 3 & 추천 이유
        - 향후 6~12개월 기준 {region} 상권에서 유망하다고 판단되는 업종 3개를 선정하라.
        - 각 업종마다:
          - 추천 업종명
          - 추천 이유 (수요, 경쟁도, 트렌드, 소비자 특성 근거)
          - 실제로 어떤 콘셉트로 풀면 좋을지 간단 제안

        ## 9. 종합 결론
        - 상권의 중장기 방향성 요약
        - 사업자/브랜드/창업자에게 제시하는 한 줄 전략 정리

        -----------------------------------------
        [참고용 RAG 문서 요약]
        {docs_text}
        -----------------------------------------

        위 구조에 따라 하나의 Markdown 문서로만 답변하라.
        """
    )

    chain = prompt | llm | StrOutputParser()
    report = chain.invoke(
        {
            "region": region,
            "industry": industry,
            "period": period,
            "docs_text": docs_text,
        }
    )

    print("\n\n===== 생성된 상권 BI 트렌드 분석 보고서 =====\n")
    print(report[:500] + "\n... (중략) ...\n")  # 콘솔에는 앞부분만

    state["report"] = report
    state["messages"].append(AIMessage(content=report))
    return state


# 5. LangGraph 구성

graph_builder = StateGraph(State)

graph_builder.add_node("parse_query", parse_query)
graph_builder.add_node("web_search", web_search_node)
graph_builder.add_node("vector_search", vector_search_node)
graph_builder.add_node("content_strategist", content_strategist_node)

graph_builder.add_edge(START, "parse_query")
graph_builder.add_edge("parse_query", "web_search")
graph_builder.add_edge("web_search", "vector_search")
graph_builder.add_edge("vector_search", "content_strategist")
graph_builder.add_edge("content_strategist", END)

graph = graph_builder.compile()


# 6. 콘솔 테스트용 메인 루프 (선택사항)

if __name__ == "__main__":
    print("🚀 상권 BI 트렌드 분석 콘솔 버전입니다. 종료: exit / q / quit\n")
    while True:
        user_input = input("\nUser    : ").strip()
        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break

        init_state: State = {
            "messages": [
                SystemMessage(
                    f"너는 상권 BI 트렌드 분석 보고서를 작성하는 데이터 분석 팀이다. "
                    f"현재 시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다."
                ),
                HumanMessage(user_input),
            ],
            "references": {"queries": [], "docs": []},
            "user_request": {},
            "report": None,
        }

        final_state = graph.invoke(init_state)
        report = final_state.get("report")

        if report:
            print("\n\n====== 최종 보고서 ======\n")
            print(report)
        else:
            print("❌ 보고서를 생성하지 못했습니다.")


# 7. Streamlit 등에서 호출할 함수

def run_analysis(user_input: str) -> str:
    """외부(UI)에서 호출하는 단일 분석 함수."""
    init_state: State = {
        "messages": [
            SystemMessage(
                f"너는 상권 BI 트렌드 분석 보고서를 작성하는 데이터 분석 팀이다. "
                f"현재 시각은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}이다."
            ),
            HumanMessage(user_input),
        ],
        "references": {"queries": [], "docs": []},
        "user_request": {},
        "report": None,
    }

    final_state = graph.invoke(init_state)
    report = final_state.get("report")

    if report:
        return report

    # 혹시 report 키가 비어있으면 messages에서 다시 찾아보기
    for m in final_state["messages"]:
        if isinstance(m, AIMessage):
            txt = m.content or ""
            if txt.strip().startswith("# "):
                return txt

    return "❌ 보고서를 생성하지 못했습니다. (보고서 텍스트를 찾지 못했어요)"
