import random
from typing import Type, Union

from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from config.custom_logger import setup_logger

logger = setup_logger(__file__)


class RandomInput(BaseModel):
    min_value: int = Field(default=1, description="乱数の最小値")
    max_value: int = Field(default=10000, description="乱数の最大値")


class RandomTool(BaseTool):
    # ★ Pydantic v2 では上書きフィールドに型注釈が必須
    name: str = "random_tool"
    description: str = (
        "指定された範囲内の整数乱数を生成します。min_value と max_value を受け取ります。"
    )
    args_schema: Type[BaseModel] = (
        RandomInput  # または: type[RandomInput] = RandomInput
    )

    # ★ Pydantic管理外の内部状態は PrivateAttr を使う
    _rng: random.Random = PrivateAttr(default_factory=random.Random)

    def _run(self, min_value: int = 1, max_value: int = 10000) -> Union[int, str]:
        """
        乱数生成ツールのメイン実行関数。

        指定された範囲内で整数の乱数を生成する。
        範囲の妥当性をチェックし、エラーが発生した場合は
        エラーメッセージを返す。

        Args:
            min_value (int, optional): 乱数の最小値。デフォルトは1
            max_value (int, optional): 乱数の最大値。デフォルトは10000

        Returns:
            Union[int, str]: 生成された乱数、またはエラー時はエラーメッセージ文字列

        Raises:
            ValueError: 最小値が最大値より大きい場合
        """
        logger.info("random_tool class start")
        logger.info(f"Generating random number between {min_value} and {max_value}")
        try:
            if min_value > max_value:
                raise ValueError("最小値は最大値以下である必要があります")
            result = self._rng.randint(min_value, max_value)
            logger.info(f"random_tool class result: {result}")
            logger.info("random_tool class end")
            return result
        except Exception as e:
            error_msg = f"乱数生成エラー: {str(e)}"
            logger.error(error_msg)
            return error_msg
