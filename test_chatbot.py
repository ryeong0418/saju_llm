import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

def load_document(file_path):
    """문서를 로드하고 청크로 분할합니다."""
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks):
    """문서 청크로부터 벡터 저장소를 생성합니다."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def create_chatbot(vector_store):
    """챗봇을 생성합니다."""
    llm = ChatOpenAI(
        model_name="gpt-4-turbo-preview",
        temperature=0.7
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    
    return chain

def main():
    # 문서 로드
    document_path = "documents/saju_profile.txt"
    chunks = load_document(document_path)
    
    # 벡터 저장소 생성
    vector_store = create_vector_store(chunks)
    
    # 챗봇 생성
    chatbot = create_chatbot(vector_store)
    
    print("사주 기반 심리 상담 챗봇이 시작되었습니다. 종료하려면 'quit'를 입력하세요.")
    
    while True:
        user_input = input("\n질문을 입력하세요: ")
        if user_input.lower() == 'quit':
            break
            
        response = chatbot({"question": user_input})
        print("\n답변:", response['answer'])

if __name__ == "__main__":
    main() 