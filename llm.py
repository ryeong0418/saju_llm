# app 시작 시 초기화 (최상단에서 1회만 호출)
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
import pinecone
import os
from langchain import hub
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class RAGChatbot():

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o")
        self.retriever = self.get_retriever()
        self.qa_chain = self.get_chain()
        self.store = {}


    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    def get_retriever(self):
        # 1️⃣ API KEY 설정
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )

        # 2️⃣ Index 이름 설정
        index_name = "saju-llm-index"

        # 3️⃣ Embedding 모델
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # 4️⃣ Vectorstore 선언
        vectorstore = PineconeVectorStore(
            index_name=index_name,
            embedding=embedding_model
        )

        # 5️⃣ Retriever 변환
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        return retriever

    def get_chain(self):
        # prompt = hub.pull("rlm/rag-prompt")
        # qa_chain = RetrievalQA.from_chain_type(
        #     llm = self.llm,
        #     retriever= self.retriever,
        #     chain_type_kwargs={"prompt": prompt}
        # )

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history,"
            "formulate a standalone question which can be understood"
            "without the chat history. Do NOT answer the question"
            "just reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        system_prompt=(
            "You are an assistant for question-answering tasks."
            "Use the following pieces of retrieved context to answer"
            "the question. If you don't know the answer, say that you"
            "don't know. Use three sentences maximum and keep the"
            "answer concise."
            "\n\n"
            "{context}"

        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            lambda session_id: self.get_session_history(session_id),
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

        return conversational_rag_chain

    def get_ai_message(self, user_message, session_id="default-session"):
        ai_message = self.qa_chain.invoke(
            {"input": user_message},
            config={"configurable": {"session_id": session_id}}
            )

        return ai_message