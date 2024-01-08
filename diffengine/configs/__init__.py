# flake8: noqa: PTH122,PTH120
# Copied from xtuner.configs.__init__
import os


def get_cfgs_name_path() -> dict:
    path = os.path.dirname(__file__)
    mapping = {}
    for root, _, files in os.walk(path):
        # Skip if it is a base config
        if "_base_" in root:
            continue
        for file_ in files:
            if file_.endswith(
                (".py", ".json"),
            ) and not file_.startswith(".") and not file_.startswith("_"):
                mapping[os.path.splitext(file_)[0]] = os.path.join(root, file_)
    return mapping


cfgs_name_path = get_cfgs_name_path()

__all__ = ["cfgs_name_path"]
