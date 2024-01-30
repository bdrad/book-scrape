import json
import re
from collections import defaultdict
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import List, Set, Tuple

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

TRACKER_MODES = {"startup", "header", "body"}
LABEL_MODES = {"Title", "Text"}
# JOINER_MODES = LABEL_MODES.union("startup")

CLEAN_MERGE = True  # ensure there is a single space between existing text and the new entry


@dataclass
class TrackerEntry:
    pg_num: InitVar[int]
    label_type: LABEL_MODES
    label_num: InitVar[int]
    txt: str
    fonts: list[list[str, float], int]  # list of `[font, size], freq`
    pgs: List[int] = field(init=False)
    labels: List[int] = field(init=False)

    def __post_init__(self, pg_num, label_num):
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
                if type(entry["fonts"]) is dict:
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

    EARLY_EXIT_TXTS = ("FURTHER READING", "SUGGESTED READING")

    def __init__(self):
        self.entries = []

    def add_entry(self, pg_num, label_type, label_num, txt, fonts) -> bool:
        """
        Returns True if the chapter is complete
        """
        if label_type not in LABEL_MODES or not txt:
            return

        if len(self.entries):
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
                if len(txt) >= 5 and all([c.islower() for c in txt[:5]]):
                    # von Hippel–Lindau Disease false flag!
                    print("$$$ lower-cased Title changed to Text:\n", txt)
                    label_type = "Text"
                # TODO if font is in text fonts

            txt = process_text(txt)

            if label_type == "Title":
                if prior_entry.label_type == "Title":
                    # prior_entry.merge(TrackerEntry(pg_num, label_type, label_num, txt, fonts))
                    # return
                    pass  # don't merge headers
                else:
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
                else:
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


# label_to_mode = lambda label: "header" if label == "Title" else "body"

# class ChapterTracker:
#     skip_interruping_header = True
#     skip_font_label_mismatch = False  # issue: causes skip_interrupting_header to be ignored
#     assert (skip_interruping_header and skip_font_label_mismatch) is False
#     multifont_new_entry = False  # not implemented
#     close_rstrip = True
#     clean_merge = False  # ensure there is a single space between existing text and the new entry

#     early_exit_txts = "SUGGESTED READING"  # TODO CFG

#     def __init__(self, mode="startup", entries: List[dict] = None) -> None:
#         assert mode in TRACKER_MODES
#         # having the argument as a list mutates it; copies chapter-to-chapter entries.
#         self.entries: List[ChapterEntry] = entries or []

#         self.mode = mode
#         self.header = self.body = ""
#         self.body_font = set()
#         self.header_font = set()
#         self.label_end = self.label_start = -1
#         self.pg_end = self.pg_start = self.pg_curr = -1

#     @property
#     def current_font(self):
#         return self.header_font if self.mode == "header" else self.body_font

#     # TODO CFG
#     font_blacklist: Set[Tuple[str, float]] = set()
#     font_bodies: Set[Tuple[str, float]] = set()
#     font_headers: Set[Tuple[str, float]] = set()
#     font_tables: Set[Tuple[str, float]] = set()

#     def font_category(self, fonts: List[Tuple[str, float]]) -> str:
#         # TODO track avg font size for each
#         cnt_header = sum([font in self.font_headers for font in fonts])
#         cnt_body = sum([font in self.font_bodies for font in fonts])
#         return "body" if cnt_body > cnt_header else "header"

#     def close(self):
#         if self.body == self.header == "":
#             return
#         if self.mode != "body":
#             print(f"### Closed while in {self.mode} -- we should only close when in `body` mode")
#         if self.close_rstrip:
#             self.header, self.body = self.header.rstrip(), self.body.rstrip()
#         self.header_font, self.body_font = self.header_font or [TOK_NOFONT], self.body_font or [
#             TOK_NOFONT
#         ]
#         self.entries.append(
#             dict(
#                 label_range=[self.label_start, self.label_end],
#                 pg_range=[self.pg_start, self.pg_end],
#                 header_font=self.header_font,
#                 header=self.header,
#                 body_font=self.body_font,
#                 body=self.body,
#             )
#         )
#         self.__init__(entries=self.entries)

#     def get_entries(self) -> List[ChapterEntry]:
#         self.close()
#         return self.entries

#     def font_log(self, fonts: List[Tuple[str, float]], mode, k=4):
#         inner_font = r"^([A-Z]+)\+(\w+?)-(\w+?)$"
#         fonts = [
#             ((re.sub(inner_font, r"\2", font) if re.search(inner_font, font) else inner_font), size)
#             for font, size in fonts
#         ]

#         if mode == "header":
#             self.font_headers.union(fonts)
#         else:
#             self.font_bodies.union(fonts)

#     def font_parse(self, label_type, fonts: List[Tuple[str, float]], txt) -> bool:
#         assert label_type in LABEL_MODES
#         label_mode = "header" if label_type == "Title" else "body"

#         if (
#             len(fonts) > 1 and self.multifont_new_entry
#         ):  # and any([font in others for font in fonts])
#             raise NotImplementedError
#             return False
#         # if not (self.header_font if self.mode == 'header' else self.body_font):
#         if not self.current_font:  # ! this line breaks ?
#             if label_mode == "header":
#                 self.header_font = set([max(fonts, key=lambda f: f[1])])  # largest font
#             else:
#                 self.body_font = set([fonts[0]])  # most frequent font
#             return True
#         else:
#             if self.font_category(fonts) != label_mode:
#                 if self.skip_font_label_mismatch:
#                     self.close()
#                     self.body_font = set(fonts)
#                     self.header_font = set(fonts)
#                     self.header = TOK_FONTBREAK
#                     self.body = txt
#                     self.mode = "body"
#                 return False
#             if label_mode == "header":
#                 self.header_font.add(max(fonts, key=lambda f: f[1]))
#             else:
#                 self.body_font.add(fonts[0])

#         return True

#     def add_txt(
#         self, txt: str, label_type: str, fonts: List[Tuple[str, float]], pg_num: int, label_num: int
#     ):
#         assert label_type in LABEL_MODES
#         if not txt:
#             return

#         # if any([font in self.font_blacklist for font in fonts]):
#         #     print(f"!!! blacklisted font, skipping `{txt}`")
#         #     return

#         log_font_mode: str = None
#         # mode to which the font will be logged to
#         # only set if confident the label correctly matches, i.e. we are extending the entry

#         if txt in self.early_exit_txts:
#             self.close()
#             return

#         if label_type == "Title":
#             if self.mode == "header":
#                 txt = clip_overlap(self.header, txt)
#                 if self.clean_merge:
#                     self.header = self.header.rstrip() + " " + txt.lstrip()
#                 else:
#                     self.header += txt
#             else:
#                 if (
#                     not term_str(self.body)
#                     and self.skip_interruping_header
#                     and pg_num == self.pg_curr
#                 ):
#                     print(f">>> interrupting header, skipping `{txt}`")
#                     return
#                 txt = clip_overlap(self.body, txt)
#                 self.close()
#                 self.header = txt
#                 self.mode = "header"
#             log_font_mode = "header"

#         elif label_type == "Text":
#             if txt.endswith(TOK_SENTENCE_TERMS):
#                 # txt += (f"\n {TOK_LABEL_END} \n" if TOK_LABEL_END else '\n')
#                 txt += "\n"
#             if self.mode == "header":
#                 txt = clip_overlap(self.header, txt)
#                 self.body = txt
#                 log_font_mode = "body"
#             else:
#                 if self.mode == "body":
#                     self.log_font_mode = "body"
#                 if self.mode == "startup" and self.header == "":
#                     self.header = TOK_STARTUP
#                 else:
#                     txt = clip_overlap(self.body, txt)
#                 if not re.search(r"\s+$", self.body) and not re.search(r"^\s+", txt):
#                     txt = " " + txt  # insert space if existing body doesn't end with whitespace
#                 if self.clean_merge:
#                     self.body = self.body.rstrip() + " " + txt.lstrip()
#                 else:
#                     self.body += txt
#             self.mode = "body"

#         if self.pg_curr < pg_num:
#             self.pg_curr = pg_num

#         if self.pg_start < 0:
#             self.pg_start = pg_num
#         if self.label_start < 0:
#             self.label_start = label_num

#         self.pg_end, self.label_end = pg_num, label_num

#         if not self.font_parse(label_type, fonts, txt):
#             # this works now, but should be placed above
#             # to raise an error when the current font isn't in [body/header]_font
#             return
#         if log_font_mode:
#             self.font_log(fonts, log_font_mode)

#     def to_file(self, fname: Path):
#         assert fname.suffix == ".json"

#         with open(fname, "w", encoding="utf8") as f:
#             print(f"*** saving to {f.name}")
#             data = self.get_entries()
#             data = [
#                 {k: (list(v) if type(v) is set else v) for k, v in entry.items()} for entry in data
#             ]  # json-ify: tuples -> lists

#             # json.dump(data, f, indent=2, ensure_ascii=False)
#             s = json.dumps(data, indent=2, ensure_ascii=False)
#             s = compact_json(s)
#             return f.write(s)
