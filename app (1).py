import os
import json
import re
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="건축 답사 여행 계획 AI 에이전트",
    page_icon="🏛️",
    layout="wide"
)

def get_secret_value(name, default=None):
    try:
        return st.secrets.get(name, os.environ.get(name, default))
    except Exception:
        return os.environ.get(name, default)


MODEL = get_secret_value("OPENAI_MODEL", "gpt-4o-mini")


def get_client():
    api_key = get_secret_value("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY가 없습니다. Streamlit Cloud의 Secrets 설정을 확인해 주세요.")
        st.stop()
    return OpenAI(api_key=api_key)


def extract_json(text):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("AI 응답을 JSON으로 해석하지 못했습니다.")


def ask_json(system_prompt, user_prompt, temperature=0.2):
    client = get_client()

    response = client.chat.completions.create(
        model=MODEL,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt + "\n\n반드시 JSON object 형식으로만 응답하세요."
            },
            {
                "role": "user",
                "content": user_prompt + "\n\n응답은 반드시 JSON object 하나만 출력하세요."
            }
        ],
        response_format={"type": "json_object"}
    )

    return extract_json(response.choices[0].message.content)


def init_state():
    defaults = {
        "stage": "input",
        "city": "",
        "days": 3,
        "candidates": [],
        "architect_chips": [],
        "style_chips": [],
        "type_chips": [],
        "final_plan": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def generate_candidates(city, days):
    system_prompt = """
당신은 건축 답사 여행 계획 AI 에이전트입니다.

이 에이전트의 핵심 원칙:
1. 사용자의 지식 수준이 후보를 제한하면 안 됩니다.
2. 후보는 대중적 유명세가 아니라 건축적 중요도를 기준으로 선정합니다.
3. 건축상 수상, 건축 전문 매체 게재, 건축사적 의미, 프리츠커상 수상 건축가의 작품, 도시사적 의미 등을 기준으로 삼습니다.
4. 유명 관광지뿐 아니라 대중은 잘 모르지만 건축적으로 중요한 숨은 명작을 포함합니다.
5. 후보에서 실제로 등장한 건축가, 시대/양식, 건물 유형을 뽑아 사용자가 선택할 수 있는 칩으로 만들 수 있게 합니다.
"""

    user_prompt = f"""
도시: {city}
여행 기간: {days}일

아래 JSON 구조로 답하세요.

{{
  "candidates": [
    {{
      "name": "건물명",
      "architect": "건축가. 모르면 미상",
      "year": "완공연도 또는 시대",
      "style": "시대/양식/사조",
      "building_type": "건물 유형",
      "area": "도시 안에서의 위치 또는 지역",
      "importance": "상 또는 중",
      "popularity": "상 또는 중 또는 하",
      "importance_basis": "건축적으로 중요한 근거",
      "visit_reason": "답사할 이유"
    }}
  ]
}}

후보는 8개에서 10개 사이로 생성하세요.
"""

    data = ask_json(system_prompt, user_prompt)
    return data.get("candidates", [])


def generate_final_plan(city, days, candidates, selected_architects, selected_styles, selected_types, direct_buildings):
    system_prompt = """
당신은 건축 답사 여행 계획 AI 에이전트입니다.

작동 원칙:
1. Phase 1 후보 목록을 바탕으로 Phase 2 운영정보, Phase 3 동선, Phase 4 일정표를 만든다.
2. 사용자가 선택한 건축가, 양식, 유형은 필터가 아니라 앵커다.
3. 앵커에 해당하는 건물은 우선 포함하되, 그 외 건축적으로 중요한 건물도 함께 추천한다.
4. 여러 선택지는 교집합이 아니라 합집합으로 처리한다.
5. 선택이 적으면 다양한 시대와 유형을 섞고, 선택이 많으면 선택 테마의 밀도를 높인다.
6. 운영시간, 휴관일, 입장료, 예약 필요 여부를 표시한다.
7. 실시간 공식 확인이 어려울 수 있으므로 운영정보에는 반드시 '공식 확인 필요' 같은 주의 문구를 포함한다.
8. 하루 일정은 이동이 무리 없도록 가까운 지역끼리 묶는다.
9. 결과는 사용자가 바로 들고 다닐 수 있는 Day 1 / Day 2 형식으로 만든다.
"""

    user_prompt = f"""
도시: {city}
여행 기간: {days}일

Phase 1 후보 목록:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

선택한 건축가 앵커:
{selected_architects}

선택한 시대/양식 앵커:
{selected_styles}

선택한 건물 유형 앵커:
{selected_types}

직접 추가한 건물:
{direct_buildings}

아래 JSON 구조로 답하세요.

{{
  "operating_info": [
    {{
      "name": "건물명",
      "opening_hours": "운영시간",
      "closed_days": "휴관일",
      "fee": "입장료",
      "reservation": "예약 필요 여부",
      "visit_mode": "내부 관람 가능 / 외관 위주 / 사전 예약 필요 등",
      "note": "공식 확인 필요 여부"
    }}
  ],
  "schedule": [
    {{
      "day": 1,
      "theme": "이 날의 주제",
      "comment": "이 날의 답사 의도",
      "stops": [
        {{
          "time": "10:00 - 12:00",
          "name": "건물명",
          "architect": "건축가",
          "style": "양식",
          "visit_info": "관람 정보 요약",
          "why_here": "이 일정에 들어간 이유",
          "movement_to_next": "다음 장소까지 이동 안내"
        }}
      ],
      "daily_tip": "하루 팁"
    }}
  ],
  "report_markdown": "최종 보고서 전체를 마크다운 형식으로 작성"
}}

반드시 {days}일 분량의 일정을 생성하세요.
"""

    return ask_json(system_prompt, user_prompt)


init_state()

st.title("🏛️ 건축 답사 여행 계획 AI 에이전트")
st.caption("도시를 입력하면 건축적으로 중요한 답사 후보를 찾고, 선택한 취향을 앵커로 삼아 일별 동선을 만듭니다.")

st.sidebar.header("1단계: 기본 정보")
city = st.sidebar.text_input("도시", value=st.session_state.city or "로마")
days = st.sidebar.slider("여행 기간", min_value=1, max_value=7, value=int(st.session_state.days))

st.sidebar.markdown("---")
st.sidebar.caption(f"사용 모델: {MODEL}")

if st.sidebar.button("처음부터 다시 시작"):
    st.session_state.stage = "input"
    st.session_state.city = ""
    st.session_state.days = 3
    st.session_state.candidates = []
    st.session_state.architect_chips = []
    st.session_state.style_chips = []
    st.session_state.type_chips = []
    st.session_state.final_plan = None
    st.rerun()


if st.session_state.stage == "input":
    st.info("왼쪽에서 도시와 여행 기간을 정한 뒤 아래 버튼을 누르세요.")

    if st.button("🔍 Phase 1 실행: 건축 후보와 선택지 생성", type="primary"):
        with st.spinner("도시의 건축 명작 후보를 수집하는 중입니다..."):
            candidates = generate_candidates(city, days)

        st.session_state.city = city
        st.session_state.days = days
        st.session_state.candidates = candidates

        st.session_state.architect_chips = sorted({
            c.get("architect", "")
            for c in candidates
            if c.get("architect") and c.get("architect") != "미상"
        })

        st.session_state.style_chips = sorted({
            c.get("style", "")
            for c in candidates
            if c.get("style")
        })

        st.session_state.type_chips = sorted({
            c.get("building_type", "")
            for c in candidates
            if c.get("building_type")
        })

        st.session_state.stage = "chips"
        st.rerun()


elif st.session_state.stage == "chips":
    st.success(f"✅ Phase 1 완료: {st.session_state.city}의 건축 후보 {len(st.session_state.candidates)}개를 생성했습니다.")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("🏛️ Phase 1 후보 목록")

        for i, c in enumerate(st.session_state.candidates, 1):
            title = f"{i}. {c.get('name', '이름 없음')} — {c.get('architect', '미상')} / {c.get('style', '양식 미상')}"
            with st.expander(title):
                st.write(f"**완공/시대:** {c.get('year', '정보 없음')}")
                st.write(f"**유형:** {c.get('building_type', '정보 없음')}")
                st.write(f"**지역:** {c.get('area', '정보 없음')}")
                st.write(f"**건축적 중요도:** {c.get('importance', '정보 없음')}")
                st.write(f"**대중적 인지도:** {c.get('popularity', '정보 없음')}")
                st.write(f"**중요도 근거:** {c.get('importance_basis', '정보 없음')}")
                st.info(c.get("visit_reason", "방문 이유 정보 없음"))

    with right:
        st.subheader("🎯 취향 앵커 선택")
        st.caption("선택한 항목만 보는 필터가 아니라, 선택한 항목을 우선 포함하는 앵커 방식입니다.")

        selected_architects = st.multiselect("건축가", st.session_state.architect_chips)
        selected_styles = st.multiselect("시대 / 양식", st.session_state.style_chips)
        selected_types = st.multiselect("건물 유형", st.session_state.type_chips)

        direct_text = st.text_input("직접 추가할 건물", placeholder="예: MAXXI, 판테온")
        direct_buildings = [x.strip() for x in direct_text.split(",") if x.strip()]

        if st.button("🚀 Phase 2~4 실행: 최종 일정표 생성", type="primary", use_container_width=True):
            with st.spinner("운영정보 조사, 동선 구성, 최종 일정표를 생성하는 중입니다..."):
                final_plan = generate_final_plan(
                    st.session_state.city,
                    st.session_state.days,
                    st.session_state.candidates,
                    selected_architects,
                    selected_styles,
                    selected_types,
                    direct_buildings
                )

            st.session_state.final_plan = final_plan
            st.session_state.stage = "result"
            st.rerun()


elif st.session_state.stage == "result":
    final_plan = st.session_state.final_plan

    st.success("✅ 최종 건축 답사 일정표가 생성되었습니다.")

    if st.button("⬅️ 취향 선택으로 돌아가기"):
        st.session_state.stage = "chips"
        st.rerun()

    tab1, tab2, tab3 = st.tabs(["📅 일별 일정", "🕒 운영정보", "📄 보고서"])

    with tab1:
        st.header(f"📍 {st.session_state.city} {st.session_state.days}일 건축 답사 일정")

        for day in final_plan.get("schedule", []):
            st.markdown(f"## Day {day.get('day')} — {day.get('theme')}")
            st.write(day.get("comment", ""))

            for stop in day.get("stops", []):
                st.markdown(f"### {stop.get('time')} | 🏛️ {stop.get('name')}")
                st.write(f"**건축가:** {stop.get('architect')}")
                st.write(f"**양식:** {stop.get('style')}")
                st.write(f"**관람 정보:** {stop.get('visit_info')}")
                st.write(f"**배치 이유:** {stop.get('why_here')}")
                move = stop.get("movement_to_next", "")
                if move:
                    st.caption(f"➡️ 다음 이동: {move}")

            st.info(day.get("daily_tip", ""))
            st.markdown("---")

    with tab2:
        st.header("Phase 2 — 운영정보")
        st.warning("운영시간, 입장료, 예약 여부는 변동될 수 있으니 실제 답사 전 공식 홈페이지에서 다시 확인하세요.")

        for item in final_plan.get("operating_info", []):
            with st.expander(item.get("name", "건물")):
                st.write(f"**운영시간:** {item.get('opening_hours')}")
                st.write(f"**휴관일:** {item.get('closed_days')}")
                st.write(f"**입장료:** {item.get('fee')}")
                st.write(f"**예약:** {item.get('reservation')}")
                st.write(f"**관람 방식:** {item.get('visit_mode')}")
                st.write(f"**주의:** {item.get('note')}")

    with tab3:
        st.header("Phase 4 — 최종 보고서")
        report = final_plan.get("report_markdown", "")
        st.text_area("보고서 미리보기", value=report, height=500)

        st.download_button(
            label="📥 보고서 다운로드 (.md)",
            data=report,
            file_name=f"{st.session_state.city}_architecture_travel_plan.md",
            mime="text/markdown",
            use_container_width=True
        )
