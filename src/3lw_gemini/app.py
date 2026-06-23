from typing import Any
from uuid import uuid4

import constants as c
import streamlit as st
from agent import HumanInTheLoopAgent
from docutil import save_as_docx
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def show_messages(messages: list[Any]) -> None:
    for message in messages:
        if isinstance(message, HumanMessage):
            with st.chat_message(message.type):
                st.write(message.content)

        elif isinstance(message, AIMessage):
            # tool_callの場合はツールの承認を求める旨を表示
            if len(message.tool_calls) != 0:
                for tool_call in message.tool_calls:
                    with st.chat_message(message.type):
                        st.write("下記の設定で推測を開始します。")
                        st.write(
                            f"""
                            **使用ツール** : {tool_call['name']}
                            | 設定値 |
                            | ------ |
                            | {tool_call['args']} |
                            """,
                            unsafe_allow_html=True,
                        )
            else:
                with st.chat_message(message.type):
                    st.write(message.content)

                    with open(c.output_md, "w", encoding="utf-8") as f:
                        f.write(message.content)
                    save_as_docx(c.output_md, c.output_filename)

                    st.download_button(
                        label="ファイルダウンロード",
                        data=open(c.output_filename, "rb"),
                        file_name=c.output_filename,
                        key=uuid4().hex,
                    )

        elif isinstance(message, ToolMessage):
            with st.chat_message(message.type):
                with st.expander("作成されたプロンプト"):
                    # st.write("ツールの実行結果")
                    st.info(message.content)

        else:
            raise ValueError(f"Unknown message type: {type(message)}")


def app() -> None:
    st.set_page_config(page_title="3 Letter Word推測", layout="wide")
    st.header("📝3 Letter Word推測", divider="rainbow")
    st.write(
        """IBM社内外で使用されている略語(ILCなど)を含む文章を入力してください。  
        略語について解説しますので、**略語を含む文章**を入力してください。  **情報が不足していると回答は作成されません。**"""
    )
    st.info("""例1)　略語：PMP、例文：PMPの資格を取得した。　例2)　例文のみ：日本IBMの女性技術者のためのコミュニティ""")
    # st.session_stateにagentを保存
    if "agent" not in st.session_state:
        st.session_state.agent = HumanInTheLoopAgent()
    agent = st.session_state.agent
    # グラフを表示
    with st.sidebar:
        st.header("⚙️設定", divider="gray")
        st.session_state.reference_pages = st.slider("ベクトル検索の取得項目数(k)", 1, 20, 10)
        st.session_state.temperature = st.slider("temperature", 0.0, 1.5, 0.0, 0.1)

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = uuid4().hex
    thread_id = st.session_state.thread_id
    # st.write(f"thread_id: {thread_id}")
    # ユーザーの指示を受け付ける
    human_message = st.chat_input()
    if human_message:
        with st.spinner():
            agent.handle_human_message(human_message, thread_id)
    # 会話履歴を表示
    messages = agent.get_messages(thread_id)
    show_messages(messages)
    # 次がhuman_review_nodeの場合は推測開始ボタンを表示
    if agent.is_next_human_review_node(thread_id):
        approved = st.button("推測開始")
        # 承認されたらエージェントを実行
        if approved:
            with st.spinner():
                agent.handle_approve(thread_id)
            # 会話履歴を表示するためrerun
            st.rerun()


app()
