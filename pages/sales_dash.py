import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.holtwinters import ExponentialSmoothing

CSV_FILE = "data/seoul_tradar_full.csv"


# ============================================================
# 숫자 포매팅 함수
# ============================================================
def format_won(x):
    x = float(x)
    if x >= 1e8:
        return f"약 {x/1e8:.1f}억 원"
    elif x >= 1e4:
        return f"{x:,.0f}원"
    return str(x)


# ============================================================
# 변동률 해석 함수
# ============================================================
def interpret_change(val):
    if pd.isna(val):
        return "데이터 없음"
    if val > 30:
        return f"📈 크게 증가(+{val:.1f}%)"
    elif val > 0:
        return f"↗ 소폭 증가(+{val:.1f}%)"
    elif val == 0:
        return "— 변화 없음"
    elif val > -30:
        return f"↘ 소폭 감소({val:.1f}%)"
    else:
        return f"📉 크게 감소({val:.1f}%)"


# ============================================================
# CSV LOAD
# ============================================================
@st.cache_data
def load_csv():
    df = pd.read_csv(CSV_FILE, dtype=str)
    num_cols = [c for c in df.columns if c.endswith("_AMT")]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="ignore")
    df["STDR_YYQU_CD"] = df["STDR_YYQU_CD"].astype(str)
    df["year"] = df["STDR_YYQU_CD"].str[:4]
    df["quarter"] = df["STDR_YYQU_CD"].str[-1]
    return df


df_all = load_csv()


# ============================================================
# 자동 인사이트 생성
# ============================================================
def generate_insight(top_df):
    if len(top_df) == 0:
        return "데이터 없음"

    best = top_df.iloc[0]
    worst = top_df.iloc[-1]
    ratio = best["THSMON_SELNG_AMT"] / max(worst["THSMON_SELNG_AMT"], 1)

    return (
        f"✔ 최고 매출 업종: {best['SVC_INDUTY_CD_NM']} ({format_won(best['THSMON_SELNG_AMT'])})\n"
        f"✔ 최저 매출 업종: {worst['SVC_INDUTY_CD_NM']} ({format_won(worst['THSMON_SELNG_AMT'])})\n"
        f"✔ 매출 차이: 약 {ratio:.1f}배"
    )


# ============================================================
# 탭 구성
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📌 매출 TOP10",
    "📊 성별·연령대·시간대 분석",
    "📈 추이 예측",
    "🧩 기준 비교"
])


# ============================================================
# 1️⃣ 매출 TOP10 탭
# ============================================================
with tab1:
    st.subheader("매출 TOP 10 분석")

    years = sorted(df_all["year"].unique())
    year_sel = st.selectbox("년도 선택", years, index=len(years)-1)
    df_year = df_all[df_all["year"] == year_sel]

    quarters = sorted(df_year["quarter"].unique())
    quarter_sel = st.selectbox("분기 선택", quarters)
    df_sel = df_year[df_year["quarter"] == quarter_sel]

    col1, col2 = st.columns(2)
    trdar_sel = col1.selectbox("상권 선택", ["(전체)"] + sorted(df_sel["TRDAR_SE_CD_NM"].dropna().unique()))
    svc_sel = col2.selectbox("업종 선택", ["(전체)"] + sorted(df_sel["SVC_INDUTY_CD_NM"].dropna().unique()))

    # 상권 선택 시
    if trdar_sel != "(전체)":
        st.markdown(f"### 🔸 <{trdar_sel}>의 매출 TOP10")

        df_t = df_sel[df_sel["TRDAR_SE_CD_NM"] == trdar_sel]
        top10 = df_t.groupby("SVC_INDUTY_CD_NM")["THSMON_SELNG_AMT"] \
            .sum().sort_values(ascending=False).head(10).reset_index()
        top10["표시"] = top10["THSMON_SELNG_AMT"].apply(format_won)

        a, b = st.columns([1, 2])
        a.dataframe(top10)

        fig = px.bar(
            top10,
            x="THSMON_SELNG_AMT", y="SVC_INDUTY_CD_NM",
            text="표시", orientation="h"
        )
        fig.update_yaxes(autorange="reversed", title="업종명")
        fig.update_xaxes(title="매출(원)")
        fig.update_traces(textposition="outside")
        b.plotly_chart(fig, use_container_width=True)

    # 업종 선택 시
    if svc_sel != "(전체)":
        st.markdown(f"### 🔸 <{svc_sel}> 업종의 상권별 매출")

        df_s = df_sel[df_sel["SVC_INDUTY_CD_NM"] == svc_sel]
        by_trdar = df_s.groupby("TRDAR_SE_CD_NM")["THSMON_SELNG_AMT"] \
            .sum().sort_values(ascending=False).reset_index()
        by_trdar["표시"] = by_trdar["THSMON_SELNG_AMT"].apply(format_won)

        a, b = st.columns([1, 2])
        a.dataframe(by_trdar)

        fig = px.bar(
            by_trdar, x="THSMON_SELNG_AMT", y="TRDAR_SE_CD_NM",
            text="표시", orientation="h"
        )
        fig.update_yaxes(autorange="reversed", title="상권명")
        fig.update_xaxes(title="매출(원)")
        fig.update_traces(textposition="outside")
        b.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2️⃣ 성별·연령대·시간대 분석 탭
# ============================================================
with tab2:
    st.subheader("상권·업종별 매출 비교")

    trdar = st.selectbox("상권 선택", sorted(df_all["TRDAR_SE_CD_NM"].unique()))
    svc = st.selectbox("업종 선택", sorted(df_all[df_all["TRDAR_SE_CD_NM"] == trdar]["SVC_INDUTY_CD_NM"].unique()))

    df_area = df_all[(df_all["TRDAR_SE_CD_NM"] == trdar) &
                     (df_all["SVC_INDUTY_CD_NM"] == svc)]

    agg = df_area.select_dtypes(include=["number"]).sum()

    # ==========================
    # ① 성별 매출 그래프
    # ==========================
    st.markdown("### 🔸 성별 매출 비교")

    gender_df = pd.DataFrame({
        "성별": ["남성", "여성"],
        "매출": [agg["ML_SELNG_AMT"], agg["FML_SELNG_AMT"]]
    })
    gender_df["표시"] = gender_df["매출"].apply(format_won)

    fig = px.bar(
        gender_df,
        x="매출",
        y="성별",
        orientation="h",
        text="표시"
    )
    fig.update_traces(textposition="outside")
    fig.update_xaxes(title="매출(원)")
    fig.update_yaxes(title="성별")
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # ② 연령대 매출 그래프
    # ==========================
    st.markdown("### 🔸 연령대별 매출 비중")

    age_cols = [
        ("AGRDE_10_SELNG_AMT", "10대"),
        ("AGRDE_20_SELNG_AMT", "20대"),
        ("AGRDE_30_SELNG_AMT", "30대"),
        ("AGRDE_40_SELNG_AMT", "40대"),
        ("AGRDE_50_SELNG_AMT", "50대"),
        ("AGRDE_60_ABOVE_SELNG_AMT", "60대↑"),
    ]

    age_df = pd.DataFrame({"연령대": label, "매출": agg[col]} for col, label in age_cols)
    age_df["표시"] = age_df["매출"].apply(format_won)

    fig = px.pie(
        age_df,
        names="연령대",
        values="매출"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ==========================
    # ③ 시간대별 매출 그래프
    # ==========================
    st.markdown("### 🔸 시간대별 매출 그래프")

    time_cols = [
        ("TMZON_00_06_SELNG_AMT", "00~06시"),
        ("TMZON_06_11_SELNG_AMT", "06~11시"),
        ("TMZON_11_14_SELNG_AMT", "11~14시"),
        ("TMZON_14_17_SELNG_AMT", "14~17시"),
        ("TMZON_17_21_SELNG_AMT", "17~21시"),
        ("TMZON_21_24_SELNG_AMT", "21~24시"),
    ]

    time_df = pd.DataFrame({"시간대": label, "매출": agg[col]} for col, label in time_cols)
    time_df["표시"] = time_df["매출"].apply(format_won)

    fig = px.bar(
        time_df,
        x="매출",
        y="시간대",
        orientation="h",
        text="표시",
        title="시간대별 매출"
    )
    fig.update_traces(textposition="outside")
    fig.update_yaxes(autorange="reversed", title="시간대")
    fig.update_xaxes(title="매출(원)")
    st.plotly_chart(fig, use_container_width=True)



# ============================================================
# 3️⃣ 추이 예측 (기존 네 코드 그대로 — 이미 정상)
# ============================================================
with tab3:
    st.subheader("상권·업종별 미래 매출 예측")

    t3_trdar = st.selectbox("상권 선택", sorted(df_all["TRDAR_SE_CD_NM"].unique()), key="t3_trdar")
    t3_svc = st.selectbox("업종 선택", sorted(df_all[df_all["TRDAR_SE_CD_NM"] == t3_trdar]["SVC_INDUTY_CD_NM"].unique()), key="t3_svc")

    # 시계열 구성
    df_reg = df_all[(df_all["TRDAR_SE_CD_NM"] == t3_trdar) & (df_all["SVC_INDUTY_CD_NM"] == t3_svc)]
    ts = df_reg.groupby("STDR_YYQU_CD")["THSMON_SELNG_AMT"].sum().reset_index()
    ts["STDR_YYQU_CD"] = ts["STDR_YYQU_CD"].astype(str)
    ts = ts.sort_values("STDR_YYQU_CD")

    # 최근 12개 분기 사용
    RECENT_N = 12
    ts_recent = ts.tail(RECENT_N)
    y_vals = ts_recent["THSMON_SELNG_AMT"].astype(float).values

    # 이상치 완화
    y_clean = y_vals.copy()
    for i in range(1, len(y_clean)):
        prev = y_clean[i-1]
        if prev > 0:
            rate = (y_clean[i] - prev) / prev
            if rate > 0.5: y_clean[i] = prev * 1.5
            elif rate < -0.5: y_clean[i] = prev * 0.5

    y_log = np.log1p(y_clean)

    model = ExponentialSmoothing(
        y_log, trend="add", seasonal="add", seasonal_periods=4
    ).fit(optimized=True, smoothing_level=0.2, smoothing_trend=0.1, smoothing_seasonal=0.05)

    FUTURE = 12
    forecast_vals = np.clip(np.expm1(model.forecast(FUTURE)), 0, None)

    last_code = ts["STDR_YYQU_CD"].iloc[-1]
    y = int(last_code[:4])
    q = int(last_code[-1])

    future_codes = []
    for _ in range(FUTURE):
        q += 1
        if q == 5:
            y += 1
            q = 1
        future_codes.append(f"{y}{q}")

    ts["구분"] = "실제"
    ts["분기"] = ts["STDR_YYQU_CD"].apply(lambda c: f"{c[:4]}년 {c[-1]}분기")
    future_df = pd.DataFrame({
        "STDR_YYQU_CD": future_codes,
        "THSMON_SELNG_AMT": forecast_vals,
        "구분": "예측",
        "분기": [f"{c[:4]}년 {c[-1]}분기" for c in future_codes]
    })

    ts_full = pd.concat([ts, future_df])

    fig = px.line(ts_full, x="분기", y="THSMON_SELNG_AMT", markers=True, color="구분")
    fig.update_xaxes(type='category', title="분기")
    fig.update_yaxes(title="매출(원)")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 4️⃣ 기준 비교 탭
# ============================================================
with tab4:

    st.header("🧩 기준 비교 분석")

    if "compare_list" not in st.session_state:
        st.session_state.compare_list = []

    colY, colQ, colAdd = st.columns([1, 1, 1])
    with colY:
        yy = st.selectbox("년도", sorted(df_all["year"].unique()), key="mcY")
    with colQ:
        qq = st.selectbox("분기", sorted(df_all["quarter"].unique()), key="mcQ")
    with colAdd:
        if st.button("➕ 기준 추가"):
            combo = f"{yy}년 {qq}분기"
            if combo not in st.session_state.compare_list:
                st.session_state.compare_list.append(combo)

    st.subheader("📌 선택된 기준")
    remove = []
    for combo in st.session_state.compare_list:
        c1, c2 = st.columns([4, 1])
        c1.write(f"**{combo}**")
        if c2.button("❌", key=f"del{combo}"):
            remove.append(combo)

    for r in remove:
        st.session_state.compare_list.remove(r)

    if len(st.session_state.compare_list) < 2:
        st.info("📌 비교를 위해 최소 2개 기준을 추가하세요")
        st.stop()

    st.subheader("업종 선택")
    st.caption("미선택시 모든 업종을 확인할 수 있습니다.")
    all_services = sorted(df_all["SVC_INDUTY_CD_NM"].unique())
    selected_services = st.multiselect("업종 선택", all_services)

    if len(selected_services) == 0:
        selected_services = all_services

    def get_df(combo):
        y = combo.split("년")[0]
        q = combo.split(" ")[1][0]
        return df_all[(df_all["year"] == y) & (df_all["quarter"] == q)]

    if len(st.session_state.compare_list) == 2:
        comboA, comboB = st.session_state.compare_list

        dfA = get_df(comboA)
        dfB = get_df(comboB)
        dfA = dfA[dfA["SVC_INDUTY_CD_NM"].isin(selected_services)]
        dfB = dfB[dfB["SVC_INDUTY_CD_NM"].isin(selected_services)]

        grpA = dfA.groupby("SVC_INDUTY_CD_NM")["THSMON_SELNG_AMT"].sum().reset_index()
        grpB = dfB.groupby("SVC_INDUTY_CD_NM")["THSMON_SELNG_AMT"].sum().reset_index()

        grpA["표시"] = grpA["THSMON_SELNG_AMT"].apply(format_won)
        grpB["표시"] = grpB["THSMON_SELNG_AMT"].apply(format_won)

        colA, colB = st.columns(2)

        with colA:
            st.subheader(f"📌 {comboA}")
            fig = px.bar(grpA, x="THSMON_SELNG_AMT", y="SVC_INDUTY_CD_NM", text="표시",
                         orientation="h")
            fig.update_xaxes(title="매출(원)")
            fig.update_yaxes(title="업종명", autorange="reversed")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig)
            st.info(generate_insight(grpA))

        with colB:
            st.subheader(f"📌 {comboB}")
            fig = px.bar(grpB, x="THSMON_SELNG_AMT", y="SVC_INDUTY_CD_NM", text="표시",
                         orientation="h")
            fig.update_xaxes(title="매출(원)")
            fig.update_yaxes(title="업종명", autorange="reversed")
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig)
            st.info(generate_insight(grpB))

        merged = pd.merge(grpA, grpB, on="SVC_INDUTY_CD_NM",
                          suffixes=("_A", "_B")).fillna(0)

        merged["변동률(%)"] = (
            (merged["THSMON_SELNG_AMT_B"] - merged["THSMON_SELNG_AMT_A"])
            / merged["THSMON_SELNG_AMT_A"].replace(0, np.nan) * 100
        )
        merged["해석"] = merged["변동률(%)"].apply(interpret_change)

        st.subheader("📈 변동률 및 해석")
        st.dataframe(merged[["SVC_INDUTY_CD_NM", "변동률(%)", "해석"]])

        fig = px.bar(merged, x="변동률(%)", y="SVC_INDUTY_CD_NM",
                     orientation="h", text="해석")
        fig.update_xaxes(title="변동률(%)")
        fig.update_yaxes(title="업종명", autorange="reversed")
        st.plotly_chart(fig)

    else:
        st.subheader("📌 다중 기준 비교 (3개 이상)")
        cols = st.columns(len(st.session_state.compare_list))

        for i, combo in enumerate(st.session_state.compare_list):
            df_sel = get_df(combo)
            df_sel = df_sel[df_sel["SVC_INDUTY_CD_NM"].isin(selected_services)]

            grp = df_sel.groupby("SVC_INDUTY_CD_NM")["THSMON_SELNG_AMT"] \
                .sum().reset_index().sort_values("THSMON_SELNG_AMT", ascending=False)
            grp["표시"] = grp["THSMON_SELNG_AMT"].apply(format_won)

            with cols[i]:
                st.subheader(combo)
                fig = px.bar(grp, x="THSMON_SELNG_AMT", y="SVC_INDUTY_CD_NM",
                             orientation="h", text="표시")
                fig.update_xaxes(title="매출(원)")
                fig.update_yaxes(title="업종명", autorange="reversed")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig)
                st.info(generate_insight(grp))
