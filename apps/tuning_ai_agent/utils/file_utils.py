import json
from typing import Any, Dict

import yaml


def load_yaml_data(file_path: str) -> Dict[str, Any]:
    """
    YAMLファイルを読み込む関数

    Args:
        file_path: 読み込むYAMLファイルのパス

    Returns:
        読み込んだYAMLデータ
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def save_json(file_path: str, data: Dict[str, Any]):
    """
    JSONファイルを保存する関数

    Args:
        file_path: 保存先のファイルパス
        data: 保存するデータ
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def save_yaml(file_path: str, data: Dict[str, Any]):
    """
    YAMLファイルを保存する関数

    Args:
        file_path: 保存先のファイルパス
        data: 保存するデータ
    """
    with open(file_path, "w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


# def ensure_dirs(dir_path: str):
#     """
#     ディレクトリを作成する関数

#     Args:
#         dir_path: 作成するディレクトリのパス
#     """
#     Path(dir_path).mkdir(parents=True, exist_ok=True)
