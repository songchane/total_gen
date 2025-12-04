# Pydantic을 이용해서 "Task" 데이터 모델(데이터 구조) 을 정의

from pydantic import BaseModel, Field
from typing import Literal

class Task(BaseModel):
    agent: Literal[
        "content_strategist",
        "communicator",
        "web_search_agent",
        "vector_search_agent",
    ] = Field(
        ...,
        description="""
        작업을 수행하는 agent의 종류.
        - content_strategist: 상권 BI 분석 보고서의 구조/내용을 설계하고 작성한다. 
        - communicator: 사용자에게 진행상황을 보고하고, 다음 지시/질문을 받는다.
        - web_search_agent: 웹 검색(DuckDuckGo)을 통해 상권 분석에 필요한 외부 데이터를 확보한다.
        - vector_search_agent: 벡터 DB(Chroma)를 통해 상권 관련 정보를 RAG 방식으로 검색한다.
        """
    )
	
    done: bool = Field(..., description="종료 여부")
    description: str = Field(..., description="어떤 작업을 해야 하는지에 대한 설명")
	
    done_at: str = Field(..., description="할 일이 완료된 날짜와 시간")
	
    def to_dict(self):
        return {
            "agent": self.agent,
            "done": self.done,
            "description": self.description,
            "done_at": self.done_at
        }
