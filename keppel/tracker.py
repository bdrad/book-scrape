import json
from collections import defaultdict
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

from keppel.cleaning import process_text
from keppel.utils import clip_overlap, compact_json, term_str


class ChapterEntry(defaultdict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # body_font = set()
    # header_font = set()
    # label_end = label_start = -1
    # pg_end = pg_start = pg_curr = -1


# TODO CFG
TOK_LABEL_END = "[LABEL_END]"
TOK_STARTUP = "[STARTUP]"
TOK_FONTBREAK = "[FONTBREAK]"
TOK_NOFONT = ("[NOFONT]", -1)
TOK_SENTENCE_TERMS = (".", "!", "?")
TOK_SENTENCE_CONTS = (",", ":", ";")
TERM_PUNC_STR = "".join(TOK_SENTENCE_TERMS)

LABEL_MODES = {"Title", "Text"}
CLEAN_MERGE = True  # ensure there is a single space between existing text and the new entry


@dataclass
class TrackerEntry:
    pg_num: InitVar[int]
    label_type: LABEL_MODES
    label_num: InitVar[int]
    txt: str
    fonts: List[Tuple[Tuple[str, float], int]]  # list of `[font, size], freq`
    # bbox: List[float] = field(init=False)
    pgs: List[int] = field(init=False)
    labels: List[int] = field(init=False)

    def __post_init__(self, pg_num: int, label_num: int):
        self.pgs = [pg_num, pg_num]
        self.labels = [label_num, label_num]

    def __json__(self):
        return asdict(self)

    def merge(self, other: "TrackerEntry"):
        assert self.label_type == other.label_type
        if CLEAN_MERGE:
            self.txt = self.txt.rstrip() + " " + other.txt.lstrip()
        else:
            self.txt += other.txt
        self.pgs[-1] = other.pgs[-1]
        self.labels[-1] = other.labels[-1]

        # sloh, but far from the bottleneck
        if self.fonts and other.fonts:
            for [name, size], freq in other.fonts:
                # print(self.fonts)
                fonts_freqless = [tuple(a) for [a, _f] in self.fonts]
                if (name, size) not in fonts_freqless:
                    self.fonts.append([[name, size], freq])
                else:
                    idx = fonts_freqless.index((name, size))
                    self.fonts[idx][-1] += freq


class Tracker:
    entries: List[TrackerEntry]

    def to_file(self, fname: Path):
        assert fname.suffix == ".json"
        with open(fname, "w", encoding="utf8") as f:
            entries_dict = []
            for entry in self.entries:
                entry = asdict(entry)
                if isinstance(entry["fonts"], dict):
                    entry["fonts"] = [[[n, s], f] for [(n, s), f] in entry["fonts"].items()]
                entry["fonts"] = sorted(entry["fonts"], key=lambda x: x[1], reverse=True)
                # print(entry["fonts"])
                entries_dict.append(entry)
            s = json.dumps(entries_dict, indent=2, ensure_ascii=False)
            s = compact_json(s)
            return f.write(s)


class PreTracker(Tracker):
    def __init__(self):
        self.entries = []

    def add_entry(self, pg_num, label_type, label_num, txt, fonts):
        entry = TrackerEntry(pg_num, label_type, label_num, txt, fonts)
        self.entries.append(entry)
        return entry


class EarlyExitException(Exception):
    pass


class JoinTracker(Tracker):
    # TODO CFG
    SKIP_INTERRUPTING_HEADER = True

    # todo make these regex patterns, add to cfg
    EARLY_EXIT_TXTS = (
        "FURTHER READING",
        "FURTHER READINGS",
        "SUGGESTED READING",
        "SUGGESTED READINGS",
    )

    def __init__(self, cfg):
        self.entries = []
        self.cfg = cfg

    def add_entry(self, pg_num, label_type, label_num, txt, fonts) -> bool:
        """
        Returns True if the chapter is complete
        """
        if label_type:
            if label_type not in LABEL_MODES or not txt:
                return
        else:
            # pdfplumber sets label_type to None
            # use cfg font info to determine label_type
            if len(fonts) == 1:
                font = fonts[0][:-1][0]  # most common, drop freq
                if font in self.cfg.get_font("head"):
                    label_type = "Title"
                elif font in self.cfg.get_font("text"):
                    label_type = "Text"

        if not len(self.entries):
            txt = process_text(txt)
        else:
            prior_entry = self.entries[-1]
            assert prior_entry.pgs[-1] <= pg_num, (prior_entry.pgs[-1], pg_num)
            assert prior_entry.labels[-1] <= label_num or prior_entry.pgs[-1] < pg_num, (
                prior_entry.labels[-1],
                label_num,
            )

            # TODO consider merging entries (under what conditions? with same label_type?)
            txt = clip_overlap(prior_entry.txt, txt, min_overlap=8)

            if txt.upper() in self.EARLY_EXIT_TXTS:
                print(f"$$$ early exit: {txt}")
                raise EarlyExitException

            # TODO if font is bad
            if label_type == "Text":
                if txt and txt == txt.upper():
                    print("$$$ all upper-cased Text changed to Title:\n", txt)
                    label_type = "Title"
                # TODO if font is in title fonts
            elif label_type == "Title":
                if len(txt) >= (k := 5) and all([c.islower() for c in txt[:k]]):
                    # von Hippel–Lindau Disease false flag!
                    print("$$$ lower-cased Title changed to Text:\n", txt)
                    label_type = "Text"
                # TODO if font is in text fonts
            else:
                # could drop
                pass

            txt = process_text(txt)

            if label_type == "Title":
                if prior_entry.label_type == "Title":
                    # prior_entry.merge(TrackerEntry(pg_num, label_type, label_num, txt, fonts))
                    # return
                    pass  # don't merge headers
                elif prior_entry.label_type == "Text":
                    if (
                        self.SKIP_INTERRUPTING_HEADER
                        and not term_str(prior_entry.txt)
                        and pg_num == prior_entry.pgs[-1]
                    ):
                        # could also check if fonts match
                        print(f">>> interrupting header, skipping `{txt}`")
                        return
            elif label_type == "Text":
                if txt.endswith(TOK_SENTENCE_TERMS):
                    # txt += (f"\n {TOK_LABEL_END} \n" if TOK_LABEL_END else '\n')
                    txt += "\n"

                if prior_entry.label_type == "Title":
                    pass
                elif prior_entry.label_type == "Text":
                    if (fonts and prior_entry.fonts) and (
                        not fonts[0][:-1] == prior_entry.fonts[0][:-1]
                    ):
                        print("!!! font mismatch, not merging")
                        pass
                    else:
                        prior_entry.merge(TrackerEntry(pg_num, label_type, label_num, txt, fonts))
                        return

            prior_entry.txt = prior_entry.txt.rstrip()

        entry = TrackerEntry(pg_num, label_type, label_num, txt, fonts)
        self.entries.append(entry)
