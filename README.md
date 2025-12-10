# 🏙️ BizDistrict AI  
**웹 검색 기반 RAG + 상권 매출 데이터 분석을 결합한 AI 상권 분석 플랫폼**

BizDistrict AI는 
**정성 분석(웹 기반 트렌드 분석 + RAG)**과 
**정량 분석(매출 데이터 분석 + 예측)**을 
하나의 서비스로 통합한 상권 BI(Business Intelligence) 자동화 플랫폼입니다.

예비 창업자, 소상공인, 프랜차이즈 본사 등  
상권 분석이 필요한 누구나 데이터를 쉽게 해석할 수 있도록 돕습니다.

---

## 📌 Features (주요 기능)

### 🔷 1) BI 트렌드 보고서 자동 생성 (RAG 기반)
- DuckDuckGo 웹 검색 → 최신 문서 수집  
- WebBaseLoader로 텍스트 로딩  
- Chroma Vector DB 기반 RAG 검색  
- GPT가 전략 분석 포함 BI 보고서 생성  

### 🔷 2) 상권 매출 분석 대시보드
- 매출 Top10  
- 성별·연령대·시간대 매출 분석  
- 기준(년도·분기) 비교 + 자동 해석  
- Plotly 그래프 기반 시각화  

### 🔷 3) 미래 매출 예측
- Holt-Winters 시계열 모델  
- 향후 12분기 예측  

### 🔷 4) GUI 기반 사용 환경
- Streamlit UI  
- 사이드바 기반 페이지 이동  

---

## 📘 프로젝트 소개 (Description)

BizDistrict AI는 기존 상권 분석의 문제점을 해결하기 위해 만들어졌습니다:

- 정성/정량 분석이 분리되어 있음  
- 비전문가에게 데이터 해석 어려움  
- 데이터 수집·정리가 복잡함  
- 미래 매출 예측 부재  
- 기준 비교 분석 어려움  

본 서비스는 **RAG + 시계열 예측 + 대시보드 분석**을 통합하여  
상권 분석을 자동화합니다.

---

## 🛠 Installation

```bash
# 1) 저장소 클론
git clone https://github.com/AISW4GEN/total_gen.git

# 2) 패키지 설치
pip install -r requirements.txt

# 3) 환경 변수 설정 (OpenAI API Key)
# Windows
set OPENAI_API_KEY=your_api_key

# Mac/Linux
export OPENAI_API_KEY=your_api_key

# 4) 실행
streamlit run overview.py

```
5) 데이터 분석 시 사용되는 csv 파일 다운로드(용량이 커서 직접 다운로드)  
data / [seoul_tradar_full.csv](https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do)

---

## ▶ Usage (사용 방법)

### ✔ 주요 페이지
- **BI_Report_Generator** → 웹 기반 RAG 상권 보고서 생성  
- **Sales_Analysis_Dashboard** → 매출 분석 / 시각화 / 비교 / 예측  

### ✔ 보고서 생성 흐름
1. 사용자 입력  
2. DuckDuckGo 웹 검색  
3. WebBaseLoader로 문서 수집  
4. Chroma DB RAG 검색  
5. GPT가 최종 BI 보고서 작성  

### ✔ 매출 분석 기능
- 년도/분기/상권/업종 선택  
- 매출 Top10 분석  
- 성별/연령/시간대 분석  
- 기준 비교 + 자동 인사이트 제공  

---

## 📂 Project Structure

```
total_gen/
│
├── overview.py
├── pages/
│     └── BI_Report_Generator.py
│     └── Sales_Analysis_Dashboard.py
├── tools.py
├── utils.py
├── models.py
├── trend.py
│
└── data/
      └── seoul_tradar_full.csv
```

---

## 🔧 Tech Stack

### AI / NLP
- OpenAI GPT  
- LangChain (RAG / Loader / Embedding)  
- Chroma Vector DB  
- DuckDuckGo Search API  

### Data / ML
- Pandas  
- NumPy  
- Statsmodels (Holt-Winters)  

### Frontend
- Streamlit  
- Plotly  

---

## 👤 Author
Ko Youngjin
Email: dseridk003@naver.com  
GitHub: https://github.com/yjk101

Lee Chansong  
Email: lcsdct64@gmail.com  
GitHub: https://github.com/songchane
