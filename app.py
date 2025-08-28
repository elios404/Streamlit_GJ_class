import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_messages
import os


load_dotenv()

st.title("Streamlit 기본 예제")
st.write("소득세 챗봇")


if "message_list" not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])


if user_question := st.chat_input(placeholder="소득세에 대해 궁금한 것을 물어봐 주세요"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role" : "user", "content" : user_question})

    with st.spinner("답변을 생성하는 중입니다."):
        ai_response = get_ai_messages(user_message=user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
    st.session_state.message_list.append({"role" : "ai", "content" : ai_message})
