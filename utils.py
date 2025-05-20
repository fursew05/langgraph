"""
웹 검색이 제대로 동작하도록 개선된 LangGraph 코드
"""
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import StateGraph, START, END
import os
from langchain_groq import ChatGroq
import os 
import dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
dotenv.load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_grop =ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)



# 기본 URL       
default_urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


# 웹 페이지 로드 및 벡터 DB 구축 함수
def setup_vectordb(urls):
    """Set up vector database from URLs"""
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=10
    )
    doc_splits = text_splitter.split_documents(docs_list)
    
    # 벡터 DB 생성
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    return vectorstore.as_retriever()

# retriever 기능 
retriever = setup_vectordb(default_urls)

# 문서 관련성 평가를 위한 구조체
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
system = """You are a grader assessing relevance of a retrieved document to a user question.
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
# 문서 평가 설정
structured_llm_grader = llm.with_structured_output(GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])
retrieval_grader = structured_llm_grader | grade_prompt


#질문 재작성 기능 Re-write Query 
system_promt = """You are a question re-writer that converts an input question to a better version that is optimized 
    for web search. Look at the input and try to reason about the underlying semantic intent / meaning.
    
    Your task is to improve the question to maximize the chance of finding relevant information. Focus on:
    1. Clarifying ambiguous terms
    2. Adding specific keywords related to the topic
    3. Phrasing it in a way that would match informational content
    
    Keep the question concise and focused on the main topic.
    """
re_write_prompt = ChatPromptTemplate.from_messages([
        ("system", system_promt),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question that will get the best search results.",
        ),
    ])


question_rewriter = llm | re_write_prompt | StrOutputParser()

# 웹 검색 도구 설정
web_search_tool = TavilySearchResults(k=5)    

# 그래프 상태 정의
class State(TypedDict):
    """
    Represents the state of our graph.
    
    Attributes:
        question: 원래 질문
        original_question: 원래 질문 (변경되지 않음)
        documents: 검색된 문서 목록
        web_search: 웹 검색 필요 여부 플래그
        generation: LLM 생성 결과
        web_results: 웹 검색 결과
        relevance_score: 문서 관련성 점수
    """
    question: str
    original_question: str
    documents: List[Document] 
    web_search: str
    generation: str
    web_results: List[Dict[str, Any]]
    relevance_score: str

# LangGraph 함수 구현
def retrieve(state:State) -> State:
    """로컬 벡터 DB에서 문서 검색"""
    print("---RETRIEVE---")
    
    question = state["question"]
    
    # 검색 수행
    documents = retriever.get_relevant_documents(question)
    
    return State(
        question= question,
        original_question= state.get("original_question", question),
        documents= documents, 
        web_search= "No",  # 기본값은 웹 검색 없음
        web_results= state.get("web_results", []),
        relevance_score= state.get("relevance_score", "")
    )
    

def grade_documents(state:State) -> State:
    """문서의 관련성 평가"""
    print("---CHECKING DOCUMENT RELEVANT IS TO QUESTION OR NOT---")
    
    question = state["question"]
    documents = state["documents"]
    
    # 문서가 없으면 바로 웹 검색 필요
    if not documents:
        print("---NO DOCUMENTS FOUND, WEB SEARCH NEEDED---")
        return State(
        question= question,
        original_question= state.get("original_question", question),
        documents= documents, 
        web_search= "yes",  # 기본값은 웹 검색 없음
        web_results= state.get("web_results", []),
        relevance_score= "no"
            )
        
    
    # 문서 평가
    filtered_docs = []
    web_search = "No"
    
    relevant_count = 0
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
            relevant_count += 1
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    
    # 관련 문서가 너무 적으면 웹 검색 필요
    if relevant_count < 2:  # 최소 2개 이상의 관련 문서가 있어야 함
        web_search = "Yes"
        print(f"---ONLY {relevant_count} RELEVANT DOCUMENTS, WEB SEARCH NEEDED---")
    
    return State(
        question= question,
        original_question= state.get("original_question", question),
        documents= filtered_docs, 
        web_search= web_search,  
        web_results= state.get("web_results", []),
        relevance_score= "yes" if filtered_docs else "no"
            )
    

def transform_query(state:State)->State:
    """질문 재작성"""
    print("---TRANSFORM QUERY---")
    
    # 원래 질문 저장
    original_question = state.get("original_question", state["question"])
    question = state["question"]
    
    # 질문 재작성
    better_question = question_rewriter.invoke({"question": original_question})
    
    print(f"---ORIGINAL QUESTION: {question}")
    print(f"---IMPROVED QUESTION: {better_question}")
    
    return State(
        question= better_question,
        original_question= original_question,
        documents= state["documents"], 
        web_search= state["web_search"],  
        web_results= state.get("web_results", []),
        relevance_score= state.get("relevance_score", "")
            )
    
def web_search(state:State)->State:
    """웹 검색 수행"""
    print("---WEB SEARCH---")
    
    question = state["question"]
    original_question = state.get("original_question", question)
    
    # 웹 검색 수행, 결과 최대 5개
    print(f"---SEARCHING WEB FOR: {question}---")
    search_results = web_search_tool.invoke({"query": question})
    
    # 검색 결과 처리
    if not search_results:
        print("---NO WEB RESULTS FOUND---")
        return State(
        question= question,
        original_question= original_question,
        documents= state["documents"], 
        web_search= "Completed",  
        web_results= [],
        generation =  "I couldn't find relevant information about your question. Please try another query."
            )
        
    
    # 웹 검색 결과를 문서 형태로 변환
    web_docs = []
    for result in search_results:
        content = result.get("content", "")
        title = result.get("title", "")
        if content:
            metadata = {"source": result.get("url", ""), "title": title}
            doc = Document(page_content=content, metadata=metadata)
            web_docs.append(doc)
    
    print(f"---FOUND {len(web_docs)} WEB RESULTS---")
    
    # 기존 문서와 웹 검색 결과 합치기
    documents = state["documents"] + web_docs
    return State(
        question= original_question,
        original_question= original_question,
        documents= documents, 
        web_search= "Completed",  
        web_results= search_results,
        relevance_score =  "yes"
            )
    

def generate(state:State) -> State:
    """답변 생성"""
    print("---GENERATE---")
    
    question = state["original_question"] if "original_question" in state else state["question"]
    documents = state["documents"]
    
    # RAG 프롬프트 준비
    from langchain import hub
    from langchain_core.output_parsers import StrOutputParser
    
    prompt = hub.pull("rlm/rag-prompt")
    
    # 검색 결과가 없는 경우 처리
    if not documents:
        return State(
        question= question,
        original_question= state.get("original_question", question),
        documents= [], 
        web_search= state.get("web_search", "No"),  
        web_results= state.get("web_results", []),
        generation =  "I couldn't find any relevant information to answer your question."
            )
        
    
    # RAG 체인으로 답변 생성
    rag_chain = prompt | llm | StrOutputParser()
    rag_chain_groq = prompt | llm_grop | StrOutputParser()
    generation = rag_chain.invoke({"context": documents, "question": question})
    # generation_groq = rag_chain_groq.invoke({"context": documents, "question": question})
    # 웹 검색 결과 사용 여부 표시
    if state.get("web_search") == "Completed" and state.get("web_results"):
        sources = []
        for idx, result in enumerate(state.get("web_results", [])[:3], 1):
            if "url" in result:
                sources.append(f"{idx}. {result.get('title', 'Source ' + str(idx))}: {result['url']}")
        
        if sources:
            generation += "\n\nSources:\n" + "\n".join(sources)
            
    return State(
        question= question,
        original_question= state.get("original_question", question),
        documents= documents, 
        web_search= state.get("web_search", "No"),  
        web_results= state.get("web_results", []),
        generation =  generation
            )
    

def decide_to_generate(state:State) -> State:
    """다음 작업 결정"""
    print("---ASSESS GRADED DOCUMENTS---")
    
    web_search = state.get("web_search", "No")
    
    if web_search == "Yes":
        # 문서가 관련이 없으면 질문 변환
        print("---DECISION: DOCUMENTS NOT RELEVANT, TRANSFORM QUERY---")
        return "질문 재작성"
    elif web_search == "Completed":
        # 웹 검색이 완료되었으면 생성
        print("---DECISION: WEB SEARCH COMPLETED, GENERATE---")
        return "결과 조합"
    else:
        # 관련 문서가 있으면 생성
        print("---DECISION: GENERATE---")
        return "결과 조합"

# 설정 및 실행 함수
def setup_and_run(user_question, example_urls=None):
    # 그래프 구성
    workflow = StateGraph(State)
    
    # 노드 추가
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("transform_query", transform_query)
    workflow.add_node("web_search_node", web_search)
    
    # 엣지 추가
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", 
        decide_to_generate, 
        {
            "질문 재작성": "transform_query",
            "결과 조합": "generate",
        }
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)
    
    # 그래프 컴파일
    app = workflow.compile()
    
    # 실행
    inputs = {"question": user_question}
    result = app.invoke(inputs)
    
    # 결과 반환
    return result["generation"]
answer = setup_and_run("tell me about the LLM")
print(answer)
# 예제 실행 (런타임 환경에서 실행할 때 활성화)
if __name__ == "__main__":
    question = "tell me about the LLM"
    answer = setup_and_run(question)
    print("\n=== FINAL ANSWER ===")
    print(answer) 