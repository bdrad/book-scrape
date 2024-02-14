from collections import defaultdict
from pathlib import Path

# import yaml
from ruamel.yaml import YAML

from keppel.utils import merge_dicts

yaml = YAML()
yaml.default_flow_style = True


FONT_KINDS = ("text", "head", "bad")


class Config:
    def __init__(self, cfg_file: Path = None):
        if cfg_file is None:
            cfg_file = Path(__file__).parent / "config.yaml"
        self.cfg_file = Path(cfg_file)
        assert cfg_file.is_file(), f"Config file {cfg_file} not found"
        with open(cfg_file, "r") as yaml_file:
            data = yaml.load(yaml_file)
        self.data = data

    def _save_data(self):
        with open(self.cfg_file, "w") as yaml_file:
            yaml.dump(self.data, yaml_file)


class BookConfig(Config):
    def __init__(self, pdf_file: Path, **kwargs):
        super().__init__(**kwargs)
        self.fname = Path(pdf_file).name

        data_books = self.data["books"]
        # todo these errors could be more helpful
        if self.fname not in data_books:
            raise ValueError(f"PDF file {self.fname} not found in config file {self.cfg_file}")
        if "chapters" not in data_books[self.fname]:
            raise ValueError(
                f"Chapters not found in book config file {self.cfg_file} for {self.fname}"
            )

        book_data = data_books[self.fname]
        base_data = self.data["base"]
        assert "chapters" not in base_data, "Chapters should only be in book config"

        # self.data = {**base_data, **book_data} # fails on nested dicts
        self.data = merge_dicts(base_data, book_data)

        self.use_pdfplumber = self.data.get("detectron", {}).get("pdfplumber", False)
        self.chapters = self._parse_chapters(self.data["chapters"])

    @staticmethod
    def _parse_chapters(chapters):
        if isinstance(str, chapters) and chapters.upper() == "DIR":
            return None
        out = []
        for i, j in zip(chapters, chapters[1:]):
            if -1 in (i, j):
                continue
            out.append((i, j))
        return out

    # this is now the default format of chapters, handled in _parse_chapters
    # def chapter_range(self):
    #     return list(zip(self.chapters, self.chapters[1:]))

    def contained_chapter(self, pg_num):
        for i, (start, end) in enumerate(self.chapters):
            if (start or float("-inf")) <= pg_num < (end or float("inf")):
                return i
        return None

    def get_font(self, kind):
        assert kind in FONT_KINDS
        return self.data["fonts"][kind]

    def get_fonts(self):
        return [self.get_font(kind) for kind in FONT_KINDS]

    def write_font(self, kind, font):
        raise NotImplementedError
        # ! I think this may overwrite whole file with just book settings
        assert kind in FONT_KINDS
        if "fonts" not in self.data["books"][self.fname]:
            self.data["books"][self.fname]["fonts"] = {k: [] for k in FONT_KINDS}
        self.data["books"][self.fname]["fonts"][kind] = font
        self._save_data()

    def write_fonts(self, fonts: list):
        assert len(fonts) == 3
        for kind, font in zip(FONT_KINDS, fonts):
            self.write_font(kind, list(font))


if __name__ == "__main__":
    cfg = BookConfig("Chest - Webb - Fundamentals of Body CT (4e).pdf")
    print(cfg.chapters)
