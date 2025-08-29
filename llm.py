from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from config import answer_examples

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def get_llm(model="gpt-4o"):
    llm = ChatOpenAI(model=model)

    return llm


def get_dictionary_chain():
    llm = get_llm()
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt2 = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        사전 : {dictionary}

        질문: {{question}}

    """)
    dictionary_chain = prompt2 | llm | StrOutputParser() # 거주자로 변경된 질문 리턴

    return dictionary_chain


def get_retriever(): # DB로 부터 쿼리와 유사한 문서를 뽑는 함수
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-index'
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embedding
    )
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})

    return retriever


def get_history_retriever(): # 생략된 쿼리에서 기존 대화를 통해서 맥락에 맞는 쿼리로 만들고 그것과 유사한 문서를 찾음
    retriever = get_retriever()
    llm = get_llm()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"), # 이게 정확하게 뭐하는건가? chat_history는 어디서 가져오는 건가? 혹은 그냥 프롬프트에 넣는것? 이건 어디에 저장?
            ("human", "{input}"),
        ]
    )

    # retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    return history_aware_retriever


def get_qa_chain(): # 이 부분 뭐하는건지 확인하기, 반드시..
    retriever = get_retriever()
    llm = get_llm()
    
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )

    history_aware_retriever = get_history_retriever()
    
    system_prompt = (
        "당신은 소득세법 전문가입니다. 사용자의 소득세법에 관한 질문에 답하세요. "
        "아래에 제공된 문서를 활용해서 답변하고,"
        "답변을 알 수 없다면, 모른다고 답하세요."
        "답변을 제공할 때 '소득세법 (XX조)에 따르면' 이라고 시작하면서 답하고"
        "2~3문장 정도의 짧은 내용의 답변으로 답하라."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            few_shot_prompt,
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    ).pick('answer')

    return conversational_rag_chain


def get_ai_messages(user_message):
    dictionary_chain = get_dictionary_chain()
    qa_chain = get_qa_chain()

    tax_chain = {"input": dictionary_chain} | qa_chain
    ai_message = tax_chain.stream(
        {
            "question" : user_message,
        },
        config={
            "configurable": {"session_id": "abc123"}
        },
    )

    return ai_message