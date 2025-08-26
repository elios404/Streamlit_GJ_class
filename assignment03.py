# %% [markdown]
# # 3차 과제

# %%
from dotenv import load_dotenv
from glob import glob
from pprint import pprint #json 형식 출력에 편리
import os

# %%
load_dotenv()

# %% [markdown]
# ## 진행순서
# 
# 1. samsung_KR.txt, skHynix_KR.txt 를 임베딩하여 vector DB 만들기
# 2. context 와 query를 위한 체인 만들기
#    1. context를 불러오고 이를 join을 통해서 하나의 문자열로 합치기
#    2. query는 RunnablePassthrough 사용하기
# 3. promptTemplate을 이용해서 위에서 context, query를 입력받아 prompt 완성하기
# 4. llm 객체를 생성하고 prompt를 input으로 넣기
# 5. StrOutputParser를 생성하고 LLM 답변을 넣어서 문자열만 반환받기

# %%
# 1. 임베딩하고 vector DB 만들기

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from tqdm import tqdm

# 1.1 문서 불러오기
text_files = glob(os.path.join('data','*_kr.txt'))

data = []

for text_file in tqdm(text_files):
    loader = TextLoader(text_file, encoding='utf-8')
    data += loader.load()

# 1.2 적당한 길이의 청크로 나누기
text_splitter = CharacterTextSplitter(
    chunk_size = 250,
    chunk_overlap = 50,
    separator='\n',
)

texts = text_splitter.split_documents(data)

# 1.3 임베딩 모델
embeddings = OpenAIEmbeddings(
    model = 'text-embedding-3-small',
)

# 1.4 vector DB 로 저장하기
vectorstore = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    collection_name="chroma_assignment",
    persist_directory='./chroma_db'
)

# %%
# 2. context, query 생성
from langchain_core.runnables import RunnablePassthrough

# 2.1 context 생성
retriever = vectorstore.as_retriever(search_kwargs={'k':2})

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

retriever_chain = retriever | format_docs

# 2.2 query
# RunnablePassthrough 사용

# %%
# 3. prompt 템플릿 만들고 context, query 채우기

from langchain.prompts import ChatPromptTemplate

template = """다음 context 만을 바탕으로 질문에 답하라.
외부 정보나 지식을 사용하지 말라.
답변이 context 와 맞지 않는 경우 혹은 일치하지 않는 경우 답변을 "잘 모르겠습니다." 라고 하라.

[Context]
{context}

[Question]
{question}

[Answer]
"""

prompt = ChatPromptTemplate.from_template(template=template)

# %%
# 4. LLM 모델 만들기

from langchain_openai import ChatOpenAI

llm = ChatOpenAI( 
    model = 'gpt-4o-mini',
    temperature = 0,
    max_tokens = 250,
)

# %%
# 5. Parser 사용하기

from langchain_core.output_parsers import StrOutputParser

# %%
# 6. chain 완성하기

chain = (
    {"context" : retriever_chain, "question" : RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "삼성은 어떤 회사이니? 그리고 하이닉스는 어떤 회사이니?"

response = chain.invoke(query)
print(response)


