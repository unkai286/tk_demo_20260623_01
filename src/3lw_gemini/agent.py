import os
from typing import Any, Literal

import constants as c
import pandas as pd
import streamlit as st
import tabulate
from dotenv import load_dotenv
from hybrid_search import HybridSearchService
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.pregel.types import StateSnapshot
from util import time_decorator

load_dotenv()
hybrid_search = HybridSearchService("three-letter-words")


@tool
def three_word_search(
    sentence: str,
    abbreviation: str = "XXXXX",
) -> str:
    """
    IBMやIT業界が使用する略語(3～5文字のアルファベット)を含む文章を解析するツール。
    ツールは1つの質問につき、1回だけ実行してください。
    Parameters
    ----------
    sentence : str
        IBMやIT業界が使用する略語を含む文章
    abbreviation : str
        3～5文字のアルファベット(指定がない場合はXXXXXとする)
    """

    # 略語検索の結果
    key_search_result = (
        hybrid_search.table.search()
        .where(f"collection_name = 'three-letter-words' and upper(abbreviations) LIKE '%{abbreviation.upper()}%'")
        .to_pandas()
    )
    key_search_result["_distance"] = 0.0

    # ベクトル検索の結果
    vector_search_result = hybrid_search.search(
        query=sentence, collection_name="three-letter-words", query_type="vector", limit=st.session_state.reference_pages
    )
    # カラムリスト
    column_key = ["abbreviations", "officialnames", "descriptions"]
    column_list = column_key + ["_distance"]

    # key列を基準にしてデータフレームをマージする
    merged_df = pd.merge(
        key_search_result[column_list],
        vector_search_result[column_list],
        on=column_list,
        how="outer",
    )

    # "_distance"の大きい順にソートし、同じキーに対して"_distance"が最大の行を保持する
    merged_df = (
        merged_df.sort_values(by="_distance", ascending=True).groupby(column_key).first().sort_values(by="_distance", ascending=True).reset_index()
    )

    result = tabulate.tabulate(
        merged_df.fillna(" ").to_records(index=False), tablefmt="github", stralign="left", numalign="left", headers=column_list
    )

    query = hybrid_search.create_prompt(
        c.prompt_filename,
        {
            "sentence": sentence,
            "abbreviation": abbreviation,
            "result": result,
        },
    )
    return query


class HumanInTheLoopAgentState(MessagesState):
    """Simple state."""


class HumanInTheLoopAgent:
    def __init__(self) -> None:
        builder = StateGraph(HumanInTheLoopAgentState)
        builder.add_node("call_llm", self._call_llm)
        builder.add_node("run_tool", self._run_tool)
        builder.add_node("human_review_node", self._human_review_node)
        builder.add_edge(START, "call_llm")
        builder.add_conditional_edges("call_llm", self._route_after_llm)
        builder.add_conditional_edges("human_review_node", self._route_after_human)
        builder.add_edge("run_tool", "call_llm")
        memory = MemorySaver()
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=["human_review_node"],
        )
        self.graph.get_graph().print_ascii()

    @time_decorator
    def _call_llm(self, state: dict) -> dict:
        # model = ChatOllama(base_url=OLLAMA_BASE_URL, model="llama3.2:3b", temperature=0).bind_tools([three_word_search])
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            # other params...
        ).bind_tools([three_word_search])
        # system_message = {
        #     "role": "system",
        #     "content": "ITや所属に関連する略語(3～5文字のアルファベット)を含む文章に関する質問以外は答えてはいけません。",
        # }
        # messages = [system_message] + state["messages"]
        messages = state["messages"]
        response = model.invoke(input=messages)
        return {"messages": [response]}  # Ollamaの応答から適切な部分を返す

    def _human_review_node(self, state: dict) -> None:
        pass

    def _run_tool(self, state: dict) -> dict:
        new_messages = []
        tools = {"three_word_search": three_word_search}
        tool_calls = state["messages"][-1].tool_calls
        for tool_call in tool_calls:
            tool = tools[tool_call["name"]]
            result = tool.invoke(tool_call["args"])
            new_messages.append(
                {
                    "role": "tool",
                    "name": tool_call["name"],
                    "content": result,
                    "tool_call_id": tool_call["id"],
                }
            )
        return {"messages": new_messages}

    def _route_after_llm(self, state: dict) -> Literal[END, "human_review_node"]:
        if len(state["messages"][-1].tool_calls) == 0:
            return END
        else:
            return "human_review_node"

    def _route_after_human(self, state: dict) -> Literal["run_tool", "call_llm"]:
        if isinstance(state["messages"][-1], AIMessage):
            return "run_tool"
        else:
            return "call_llm"

    def handle_human_message(self, human_message: str, thread_id: str) -> None:
        # 承認待ちの状態でhuman_messageが送信されるのは、ツールの呼び出しを修正したい状況
        # そのため、次がhuman_review_nodeの場合、ツールの呼び出しが失敗したことをStateに追加
        # 参考: https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/review-tool-calls/#give-feedback-to-a-tool-call
        if self.is_next_human_review_node(thread_id):
            last_message = self.get_messages(thread_id)[-1]
            tool_reject_message = ToolMessage(
                content="Tool call rejected",
                status="error",
                name=last_message.tool_calls[0]["name"],
                tool_call_id=last_message.tool_calls[0]["id"],
            )
            self.graph.update_state(
                config=self._config(thread_id),
                values={"messages": [tool_reject_message]},
                as_node="human_review_node",
            )

        for _ in self.graph.stream(
            input={"messages": [HumanMessage(content=human_message)]},
            config=self._config(thread_id),
            stream_mode="values",
        ):
            pass

    def handle_approve(self, thread_id: str) -> None:
        for _ in self.graph.stream(
            input=None,
            config=self._config(thread_id),
            stream_mode="values",
        ):
            pass

    def get_messages(self, thread_id: str) -> Any:
        if "messages" in self._get_state(thread_id).values:
            return self._get_state(thread_id).values["messages"]
        else:
            return []  # noqa: PD011

    def is_next_human_review_node(self, thread_id: str) -> bool:
        graph_next = self._get_state(thread_id).next
        return len(graph_next) != 0 and graph_next[0] == "human_review_node"

    def _get_state(self, thread_id: str) -> StateSnapshot:
        return self.graph.get_state(config=self._config(thread_id))

    def _config(self, thread_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": thread_id}}

    def mermaid_png(self) -> bytes:
        return self.graph.get_graph().draw_mermaid_png()
