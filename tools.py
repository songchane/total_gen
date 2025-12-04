# 웹 검색 + 웹 문서 수집 + 벡터 DB(RAG) 구축 + 벡터 검색 / 전체 파이프라인을 하나로 구성한 RAG(Web-RAG) 모듈

from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import json
import os

from dotenv import load_dotenv
load_dotenv()


absolute_path = os.path.abspath(__file__)  # 현재 파일의 절대 경로
current_path = os.path.dirname(absolute_path)  # 현재 .py 파일이 있는 폴더 경로

# RAG를 위한 설정
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 오픈AI Embedding 설정
embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# 크로마 DB 저장 경로 설정
persist_directory = f"{current_path}/data/chroma_store"

# Chroma 객체 생성
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

@tool
def web_search(query: str):
    """
    DuckDuckGo 기반으로 주어진 query에 대해 웹검색을 하고, 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        (list[dict], str): 검색 결과 리스트와 JSON 파일 경로
    """
    results = []

    # DuckDuckGo 검색
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=5):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "content": r.get("body", ""),
                    "raw_content": None,
                }
            )

    # raw_content가 없으면 직접 페이지를 로드해서 채우기
    for result in results:
        if not result["raw_content"]:
            url = result.get("url")
            if not url:
                result["raw_content"] = result["content"]
                continue

            try:
                result["raw_content"] = load_web_page(url)
            except Exception as e:
                print(f"Error loading page: {url}")
                print(e)
                result["raw_content"] = result["content"]

    # 결과를 JSON으로 저장
    os.makedirs(f"{current_path}/data", exist_ok=True)
    resources_json_path = f"{current_path}/data/resources_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.json"
    with open(resources_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results, resources_json_path  # 검색 결과와 JSON 파일 경로 반환


def web_page_to_document(web_page):
    # raw_content와 content 중 정보가 많은 것을 page_content로 한다.
    raw = web_page.get("raw_content") or ""
    content = web_page.get("content") or ""

    if len(raw) > len(content):
        page_content = raw
    else:
        page_content = content

    # 랭체인 Document로 변환
    document = Document(
        page_content=page_content,
        metadata={
            "title": web_page.get("title", ""),
            "source": web_page.get("url", ""),
        },
    )

    return document


def web_page_json_to_documents(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        resources = json.load(f)

    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print("Splitting documents...")
    print(f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할합니다.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(documents)

    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits


# documents를 chroma DB에 저장하는 함수
def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    print("Documents를 Chroma DB에 저장합니다.")

    if not documents:
        print("문서가 비어 있습니다.")
        return

    # documents의 url 가져오기
    urls = [document.metadata.get("source", "") for document in documents]

    # 이미 vectorstore에 저장된 urls 가져오기
    stored = vectorstore._collection.get()
    stored_metadatas = stored.get("metadatas", []) if stored else []
    stored_web_urls = [metadata.get("source", "") for metadata in stored_metadatas]

    # 새로운 urls만 남기기
    new_urls = set(urls) - set(stored_web_urls)

    # 새로운 urls에 대한 documents만 남기기
    new_documents = []

    for document in documents:
        if document.metadata.get("source", "") in new_urls:
            new_documents.append(document)
            print("New document:", document.metadata)

    # 새로운 documents를 Chroma DB에 저장
    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 크로마 DB에 저장
    if splits:
        vectorstore.add_documents(splits)
    else:
        print("No new urls to process")


# json 파일에서 documents를 만들고, 그 documents들을 Chroma DB에 저장
def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def load_web_page(url: str):
    loader = WebBaseLoader(url, verify_ssl=False)

    content = loader.load()
    raw_content = content[0].page_content.strip()

    while "\n\n\n" in raw_content or "\t\t\t" in raw_content:
        raw_content = raw_content.replace("\n\n\n", "\n\n")
        raw_content = raw_content.replace("\t\t\t", "\t\t")

    return raw_content


@tool
def retrieve(query: str, top_k: int = 5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs


if __name__ == "__main__":
    # 간단한 동작 테스트용 
    # results, resources_json_path = web_search.invoke({"query": "서울 상권 분석 30대 매출"})
    # print(results)

    # retrieved_docs = retrieve.invoke({"query": "성수동 카페 상권 매출 분석"})
    # print(retrieved_docs)
    pass
