from pathlib import Path

def image_list(path: Path):
    return list(path.rglob("*jpeg"))

def sum_list(data_list):
    return [sum(data_list)]