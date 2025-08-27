# %%
from dotenv import load_dotenv
from glob import glob
import os

load_dotenv()

# %% [markdown]
# # 내가 작성한 코드

# %% [markdown]
# ## 1. 문서 로딩

# %%
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader('data/tax.docx')

tax_doc = loader.load()

print(tax_doc[0].page_content)

# %% [markdown]
# ## 2. 문서 Chunk 로 자르기

# %%
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100,
    separators=['\n\n']
)

chunked_docs = text_splitter.split_documents(tax_doc)

print(list(len(doc.page_content) for doc in chunked_docs)[:20])
print(len(chunked_docs))

# %%
chunked_docs[231].page_content

# %% [markdown]
# ## 3. 임베딩 모델 가져오기

# %%
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name = "BAAI/bge-m3")

# %% [markdown]
# ## 4. 벡터 DB 구축

# %%
from langchain_chroma import Chroma

chroma_db = Chroma.from_documents(
    documents=chunked_docs,
    embedding=embedding_model,
    collection_name='tax_project',
    persist_directory='./tax_db',
)

# %% [markdown]
# ## 5. Retriever 구성하기

# %%
from langchain_community.utils.math import cosine_similarity

retriever = chroma_db.as_retriever(
    # search_type = "similarity_score_threshold",
    search_kwargs={
        'k' : 3,
        # "score_threshold" : 0.6,
    },
)

query = "신탁재산 귀속 소득에 대한 납세의무의 범위는 소득세법에서 몇 조 몇 번이니?"
retriever_docs = retriever.invoke(query)

print(f"찾은 문서 갯수 : {len(retriever_docs)}")

print(f"쿼리 : {query}")
print("검색 결과 : ")
for doc in retriever_docs:
    score = cosine_similarity(
        [embedding_model.embed_query(query)],
        [embedding_model.embed_query(doc.page_content)]
    )[0][0]
    print(f" - {doc.page_content} \n [유사도 : {score:.4f}]")
    print("="*30)

# %% [markdown]
# ## 6. 프롬프트 체인 구성 (Prompt + Chain)

# %%
from langchain.prompts import ChatPromptTemplate

template="""당신은 세금 전문 챗봇입니다. 다음 정보를 바탕으로 질문에 답변하세
요. 반드시 context에 있는 내용을 기반으로 답하고 정보를 관련된 정보가 아예 없다는 가정 하에만 다른 정보기반으로 대답해줘.
또한 이 템플릿으로 질문이 넘어올때는 답변에 "Rag기반 답변입니다." 라고 적어줘.

{context}

질문: {question}"""

prompt = ChatPromptTemplate.from_template(template = template)

# %%
def join_docs(documents):
    return "\n\n".join([doc.page_content for doc in documents])

retriever_chain = retriever | join_docs

# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model = "gpt-4o-mini",
    temperature=0,
)

# %%
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

rag_chain = (
    {"context" : retriever_chain, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# %% [markdown]
# ## 7. Gradio 기반 챗봇 UI

# %%
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
# 이걸 굳이 쓰는 이유는?

def answer_invoke(message, history):

    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    history_langchain_format.append(HumanMessage(content=message))

    response = rag_chain.invoke(message)

    final_answer = llm.invoke(
        history_langchain_format[:-1] + [AIMessage(content=response)] + [HumanMessage(content=message)]
    )

    return final_answer.content

demo = gr.ChatInterface(fn=answer_invoke, title="QA Bot")

demo.launch()

# %%
demo.close()


