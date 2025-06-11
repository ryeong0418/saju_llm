import streamlit as st
from llm import RAGChatbot

# 페이지 설정
st.set_page_config(
    page_title="사주 기반 심리 상담 챗봇",
    page_icon="🔮",
    layout="wide"
)

# 제목과 설명
st.title("🔮 사주 기반 심리 상담 챗봇")
st.markdown("""
이 챗봇은 당신의 사주 정보를 바탕으로 심리 상담과 인생 조언을 제공합니다.
자유롭게 질문해보세요!
""")

# 세션 상태 초기화
if 'chatbot' not in st.session_state:
    with st.spinner('챗봇을 초기화하는 중...'):
        st.session_state.chatbot = RAGChatbot()
        st.session_state.messages = []

# 채팅 인터페이스
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 챗봇 응답
    with st.chat_message("assistant"):
        with st.spinner('생각 중...'):
            response = st.session_state.chatbot.get_ai_message(prompt)
            st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            
            # 평가 결과 표시
            with st.expander("답변 평가 결과"):
                evaluation = response['evaluation']
                for metric_name, result in evaluation.items():
                    st.subheader(f"📊 {metric_name}")
                    
                    # 점수 표시
                    score = result['score']
                    if metric_name == "해악성":
                        score = 1 - score  # 해악성은 낮을수록 좋음
                    
                    st.write(f"점수: {score:.2f}")
                    st.progress(score)
                    
                    # 평가 이유 표시
                    st.write("평가 이유:")
                    st.info(result['reason'])

# 사이드바에 정보 추가
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    st.markdown("""
    1. 채팅창에 질문을 입력하세요
    2. 챗봇이 사주 정보를 바탕으로 답변합니다
    3. 심리 상담, 인생 조언 등 다양한 주제로 대화할 수 있습니다
    """)
    
    st.header("💡 예시 질문")
    st.markdown("""
    - 제 사주에서 가장 강한 부분은 무엇인가요?
    - 현재 직장에서의 어려움을 어떻게 극복하면 좋을까요?
    - 인간관계에서 주의해야 할 점이 있나요?
    - 앞으로의 운세는 어떤가요?
    """)