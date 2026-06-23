import os
from datetime import timedelta
from logging import INFO, WARNING, basicConfig, getLogger
from typing import List

import constants as c
import lancedb
import pandas as pd
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import RRFReranker
from util import time_decorator

logger = getLogger(__name__)
basicConfig(level=INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger.setLevel(INFO)
getLogger("httpx").setLevel(WARNING)

load_dotenv()

embeddings = get_registry().get("gemini-text").create()


class Threewords(LanceModel):
    abbreviations: str
    officialnames: str
    descriptions: str = embeddings.SourceField()
    vector: Vector(embeddings.ndims()) = embeddings.VectorField()  # type: ignore
    collection_name: str | None


class HybridSearchService:
    @time_decorator
    def __init__(self, table_name) -> None:
        self.prompt_path = c.prompt_dir
        self._llm = None
        self._db = lancedb.connect("/workspace/data/lancedb")
        self._tablename = table_name
        try:
            if table_name == "three-letter-words":
                self._table = self._db.create_table(self._tablename, schema=Threewords)
                self.load_csv()
        except Exception as e:
            logger.warning(e)
        logger.info(self.table.schema)
        logger.info(f"現在のページ数 : {self.table.count_rows()}")

    @property
    def table(self):
        self._table = self._db.open_table(self._tablename)
        return self._table

    @time_decorator
    def rebuild_index(self, fts_field_nane="text"):
        try:
            self.table.optimize(cleanup_older_than=timedelta(days=0), delete_unverified=True)
        except Exception as e:
            logger.warning(e)

        try:
            self.table.create_index(metric="cosine", vector_column_name="vector", index_type="IVF_FLAT", replace=True)
            self.table.create_fts_index(fts_field_nane, replace=True, use_tantivy=False)
            logger.info(self.table.list_indices())
        except Exception as e:
            logger.warning(e)

    @time_decorator
    def load_csv(self):
        df = pd.read_csv(c.csv_dir + c.csv_filename)
        self.load_df(df, "three-letter-words")
        self.rebuild_index()

    @time_decorator
    def load_df(self, df, collection_name, delete=True, batch_size=10):
        if delete is True:
            self.table.delete(where=f"collection_name = '{collection_name}'")

        df["collection_name"] = collection_name
        # 10件ずつバッチ処理でデータをロード
        for start in range(0, len(df), batch_size):
            batch = df.iloc[start : start + batch_size]  # データをスライス
            try:
                self.table.add(batch.to_dict(orient="records"))  # 辞書形式に変換して追加
            except Exception as e:
                logger.error(e)
            logger.info(f"{self.table.count_rows()}レコードをベクトル化しました。")

    @time_decorator
    def search(self, query, collection_name=None, limit=3, query_type="hybrid", fts_column="text"):
        if not collection_name:
            collection_name = self.collection_name

        reranker = RRFReranker(return_score="all")

        if query_type == "vector":
            results = (
                self.table.search(
                    query=query,
                    query_type=query_type,
                    vector_column_name="vector",
                )
                .where(f"collection_name = '{collection_name}'", prefilter=True)
                .limit(limit)
                .to_pandas()
            )

        if query_type == "hybrid":
            results = (
                self.table.search(
                    query=query,
                    query_type=query_type,
                    vector_column_name="vector",
                    fts_columns=fts_column,
                )
                .where(f"collection_name = '{collection_name}'", prefilter=True)
                .rerank(reranker)
                .limit(limit)
                .to_pandas()
            )
        return results

    # テンプレートファイルを基にプロンプトを作成
    def create_prompt(self, filename, render_data):
        # テンプレートの取得
        env = Environment(loader=FileSystemLoader(self.prompt_path))
        template = env.get_template(filename)
        # テンプレートの描画
        rendered = template.render(render_data)
        return str(rendered)


def main():
    pass


if __name__ == "__main__":
    main()
