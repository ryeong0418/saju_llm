# app 시작 시 초기화 (최상단에서 1회만 호출)
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from metrics import (
    SajuRelevanceMetric,
    SajuFaithfulnessMetric,
    SajuClarityMetric,
    SajuHarmMetric
)
from typing import Dict, Any


class RAGChatbot():

    def __init__(self):
        self.retriever = self.get_retriever()
        self.chain = self.get_chain()
        self.metrics = [
            ("사주 관련성", SajuRelevanceMetric(threshold=0.7)),
            ("사주 논리성", SajuFaithfulnessMetric(threshold=0.7)),
            ("명확성", SajuClarityMetric(threshold=0.7)),
            ("해악성", SajuHarmMetric(threshold=0.3))
        ]

    def get_retriever(self):
        """벡터 저장소에서 검색기(retriever)를 생성합니다."""
        # 문서 로드
        loader = TextLoader("documents/saju_profile.txt", encoding='utf-8')
        documents = loader.load()
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # 벡터 저장소 생성
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def get_chain(self):
        """대화형 RAG 체인을 생성합니다."""
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
            retriever=self.retriever,
            memory=memory
        )
        
        return chain

    def get_ai_message(self, user_question: str, session_id: str = None) -> Dict[str, Any]:
        """사용자 질문에 대한 AI 응답을 생성하고 평가합니다."""
        # AI 응답 생성
        response = self.chain.invoke({"question": user_question})
        answer = response['answer']
        
        # 응답 평가
        test_case = LLMTestCase(
            input=user_question,
            actual_output=answer,
            expected_output=None  # 사주 상담에서는 정확한 답이 없으므로 None
        )
        
        # 모든 메트릭으로 평가
        evaluation_results = {}
        for metric_name, metric in self.metrics:
            score = metric.measure(test_case)
            evaluation_results[metric_name] = {
                "score": score,
                "reason": metric.reason
            }
        
        return {
            "answer": answer,
            "evaluation": evaluation_results
        }