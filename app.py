import streamlit as st
from llm import RAGChatbot

st.set_page_config(page_title="saju", page_icon="🤖")

st. title('🐭🐔🐯🐇🐮saju')
st.caption('saju에 관련된 모든 것을 알려드립니다!')


# 1️⃣ RAGChatbot 인스턴스를 session_state에 저장 (최초 1회만 생성)
if 'rag_chatbot' not in st.session_state:
    st.session_state.rag_chatbot = RAGChatbot()

if 'session_id' not in st.session_state:
    st.session_state.session_id = 'saju-user-session'


#채팅 input은 사용자의 입력이다.

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message['role']):
        st.write(message['content'])

if user_question := st.chat_input(placeholder='saju에 관련된 궁금한 내용들을 말씀해주세요!'):
    with st.chat_message('user'):
        st.write(user_question)
    st.session_state.message_list.append({'role':'user', 'content':user_question})

    with st.spinner('답변을 생성하는 중입니다.'):
        ai_message = st.session_state.rag_chatbot.get_ai_message(user_question, session_id=st.session_state.session_id)
        with st.chat_message('ai'):
            st.write(ai_message['answer'])
        st.session_state.message_list.append({"role":"ai", "content": ai_message['answer']})