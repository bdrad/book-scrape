from pathlib import Path
from typing import List

import yaml


def extract_chapters(fname: Path) -> List[int]:
    # TODO implement this
    # - this works only iff the PDF has indexs (embedded Table of Contents)
    # - this lacks the final page of end of final chapter
    import fitz

    pdf = fitz.open(fname)

    chapters = []
    for lvl, title, pg in pdf.get_toc():
        if lvl != 1:
            continue
        # if re.search(r"chapter \d+", title, flags=re.IGNORECASE):
        # print(f"{title}\tPage: {pg}")
        chapters.append(pg - 1)  # 0-indexed
    return chapters


class Config:
    def __init__(self, cfg_file: Path = None):
        if cfg_file is None:
            cfg_file = Path(__file__).parent / "config.yaml"
        self.cfg_file = Path(cfg_file)
        assert cfg_file.is_file(), f"Config file {cfg_file} not found"
        with open(cfg_file, "r") as yaml_file:
            data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        self.data = data


class BookConfig(Config):
    def __init__(self, pdf_file: Path, **kwargs):
        super().__init__(**kwargs)
        data = self.data["books"]

        self.fname = Path(pdf_file).name
        if self.fname not in data:
            raise ValueError(f"PDF file {self.fname} not found in config file {self.cfg_file}")
        if "chapters" not in data[self.fname]:
            raise ValueError(f"Chapters not found in config file {self.cfg_file}")
        self.chapters = data[self.fname]["chapters"]

    def chapter_range(self):
        return list(zip(self.chapters, self.chapters[1:]))

    def contained_chapter(self, pg_num):
        chapter_range = self.chapter_range()
        for i, (start, end) in enumerate(chapter_range):
            if (start or float("-inf")) <= pg_num < (end or float("inf")):
                return i
        return None


if __name__ == "__main__":
    cfg = BookConfig("Chest - Webb - Fundamentals of Body CT (4e).pdf")
    print(cfg.chapters)
