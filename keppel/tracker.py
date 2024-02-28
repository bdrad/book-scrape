import json
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import List, Set, Tuple

import numpy as np

from keppel.cleaning import clean_fonts, process_text, reduce_sort_fonts
from keppel.utils import clip_overlap, compact_json, term_str

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
class Label:
    pg_num: InitVar[int]
    label_num: InitVar[int]
    label_type: str
    bbox: InitVar[Tuple[float, float, float, float]]

    txt: str = field(default=None)
    fonts: List[Tuple[Tuple[str, float], int]] = field(
        default=None
    )  # list of `((font, size), freq)`, sorted by freq

    id: int = field(init=False)
    pgs: List[int] = field(init=False)
    labels: List[int] = field(init=False)
    bbox_raw: Tuple[float, float, float, float] = field(init=False)
    bbox_pad: Tuple[float, float, float, float] = field(init=False)

    def __post_init__(self, pg_num: int, label_num: int, bbox: Tuple[float, float, float, float]):
        self.id = label_num
        self.labels = [label_num, label_num]
        self.pgs = [pg_num, pg_num]
        self.bbox_raw = bbox
        self.bbox_pad = None

    def calc_bbox_pad(self, w_im, h_im, w_pg, h_pg, pad=0.01) -> Tuple[float, float, float, float]:
        if not self.bbox_pad:
            x0, y0, x1, y1 = self.bbox_raw
            w_ratio, h_ratio = w_pg / w_im, h_pg / h_im
            x0, x1 = x0 * w_ratio, x1 * w_ratio
            y0, y1 = y0 * h_ratio, y1 * h_ratio
            pW = pad * w_im / 8
            pH = pad * h_im / 16
            x0, x1 = x0 - pW, x1 + pW
            y0, y1 = y0 - pH, y1 + pH
            self.bbox_pad = (x0, y0, x1, y1)
        return self.bbox_pad

    def __json__(self, prune: Set[str] = None):
        d = asdict(self)
        for k in prune or []:
            d.pop(k)
        return d

    def merge(self, other: "Label"):
        assert self.label_type == other.label_type
        if CLEAN_MERGE:
            self.txt = self.txt.rstrip() + " " + other.txt.lstrip()
        else:
            self.txt += other.txt
        self.pgs[-1] = other.pgs[-1]
        self.labels[-1] = other.labels[-1]

        # # sloh, but far from the bottleneck
        # if self.fonts and other.fonts:
        #     for [name, size], freq in other.fonts:
        #         # print(self.fonts)
        #         fonts_freqless = [tuple(a) for [a, _f] in self.fonts]
        #         if (name, size) not in fonts_freqless:
        #             self.fonts.append([[name, size], freq])
        #         else:
        #             idx = fonts_freqless.index((name, size))
        #             self.fonts[idx][-1] += freq

        # merge fonts properly
        if self.fonts and other.fonts:
            self.fonts = reduce_sort_fonts(self.fonts + other.fonts)
        else:
            self.fonts = other.fonts or self.fonts


class LabelTypes:
    TITLE = "title"
    FIG = "figure"
    BODY = "plain text"
    HEAD = "header"
    PG = "page number"
    FN = "footnote"
    FOOTER = "footer"
    TABLE = "table"
    MATH = "equation"
    CAP_TABLE = "table caption"
    CAP_FIG = "figure caption"
    COL_FULL = "full column"
    COL_SUB = "sub column"
    S_ALL = {
        TITLE,
        FIG,
        BODY,
        HEAD,
        PG,
        FN,
        FOOTER,
        TABLE,
        MATH,
        CAP_TABLE,
        CAP_FIG,
        COL_FULL,
        COL_SUB,
    }
    S_IGNORE = {PG, MATH}
    S_CAPTION = {CAP_TABLE, CAP_FIG}
    S_TEXT = {TITLE, BODY, HEAD, FN, FOOTER}
    S_IMG = {FIG, TABLE}


# label_type_ignore = {"page number", "equation"}
# label_type_text = {"title", "plain text", "header", "footnote", "footer"}
# label_type_caption = {"figure caption", "table caption"}
# label_type_img = {"figure", "table"}


class Encoder(json.JSONEncoder):
    # Source: https://stackoverflow.com/a/57915246/7833617
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(Encoder, self).default(obj)


class Tracker:
    def __init__(self):
        self.entries: List[Label] = []

    def add_label(self, label: Label) -> None:
        self.entries.append(label)
        return

    def to_file(self, fname: Path, prune: Set[str] = None):
        assert fname.suffix == ".json"
        with open(fname, "w", encoding="utf8") as f:
            entries_dict = []
            for entry in self.entries:
                entry = entry.__json__(prune)
                if entry["fonts"]:
                    if isinstance(entry["fonts"], dict):
                        entry["fonts"] = [[[n, s], f] for [(n, s), f] in entry["fonts"].items()]
                    # TODO
                    entry["fonts"] = sorted(entry["fonts"], key=lambda x: x[1], reverse=True)
                    # print(entry["fonts"])
                entries_dict.append(entry)
            s = json.dumps(entries_dict, indent=2, ensure_ascii=False, cls=Encoder)
            s = compact_json(s)
            return f.write(s)


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

    def add_label(self, label: Label) -> None:
        """
        Returns True if the chapter is complete
        """

        # TODO font categorization + filtering/re-assigning
        # if len(fonts) == 1:
        #     font = fonts[0][:-1][0]  # most common, drop freq
        #     if font in self.cfg.get_font("head"):
        #         label_type = "Title"
        #     elif font in self.cfg.get_font("text"):
        #         label_type = "Text"

        if label.label_type not in LabelTypes.S_ALL:
            print(f"!!! unknown label type: {label.label_type}")
            return
        if label.label_type in LabelTypes.S_IGNORE:
            return
        if label.label_type in LabelTypes.S_TEXT and not label.txt.strip():
            return
        if label.txt.upper() in self.EARLY_EXIT_TXTS:
            print(f"$$$ early exit: {label.txt}")
            raise EarlyExitException

        label.fonts = clean_fonts(label.fonts)  # sorts fonts by freq

        if label.label_type in LabelTypes.S_TEXT:
            if label.label_type != LabelTypes.HEAD and label.txt == label.txt.upper():
                print("$$$ all upper-cased non-head changed to head:\n", label.txt)
                label.label_type = LabelTypes.HEAD
            elif (
                label.label_type == LabelTypes.HEAD
                and len(label.txt) >= (k := 5)
                and all([c.islower() for c in label.txt[:k]])
            ):
                # von Hippel–Lindau Disease false flag! (for k<5)
                print("$$$ lower-cased head changed to body:\n", label.txt)
                label.label_type == LabelTypes.BODY

        if len(self.entries):
            prior_entry = self.entries[-1]
            assert prior_entry.pgs[-1] <= label.pgs[0], (prior_entry.pgs[-1], label.pgs[0])
            assert prior_entry.labels[-1] <= label.id or prior_entry.pgs[-1] < label.pgs[0], (
                prior_entry.labels[-1],
                label.id,
            )

            # TODO text overlap more generally
            label.txt = clip_overlap(prior_entry.txt, label.txt, min_overlap=8)

            # label.txt = process_text(label.txt)

            # TODO consider more ways to merge entries
            # No further re-assignments past here -- only merging
            if label.label_type == LabelTypes.BODY:
                if label.txt.endswith(TOK_SENTENCE_TERMS):
                    # label.txt += (f"\n {TOK_LABEL_END} \n" if TOK_LABEL_END else '\n')

                    label.txt += "\n"
                if prior_entry.label_type == LabelTypes.HEAD:
                    pass
                elif prior_entry.label_type == LabelTypes.BODY:
                    if (label.fonts and prior_entry.fonts) and (
                        not label.fonts[0][:-1] == prior_entry.fonts[0][:-1]
                    ):
                        print("!!! font mismatch, not merging")
                        pass
                    else:
                        prior_entry.merge(label)
                        return
            elif label.label_type == LabelTypes.HEAD:
                if prior_entry.label_type == LabelTypes.BODY:
                    if (
                        self.SKIP_INTERRUPTING_HEADER
                        and not term_str(prior_entry.txt)
                        and label.pgs[0] == prior_entry.pgs[-1]
                    ):
                        # could also check if fonts match
                        print(f">>> interrupting header, skipping `{label.txt}`")
                        return
                    # if prior_entry.label_type == "Title":
                    #     # prior_entry.merge(Label(pg_num, label_type, label_num, txt, fonts))
                    #     # return
                    #     pass  # don't merge headers
                    # elif prior_entry.label_type == "Text":

            prior_entry.txt = prior_entry.txt.rstrip()

        self.entries.append(label)
        return

    def to_file(self, fname: Path):
        super().to_file(fname, prune={"bbox_pad"})  # "id",

    # TODO overlap bbox handling
    # TODO text cleaning
    # TODO figure caption association
    # if extract_figs and (
    #     (label.label_type in label_type_caption)
    #     or (
    #         last_entry
    #         and last_entry.label_type
    #         in label_type_img  # todo make categorization a Label class property
    #         and label.label_type in label_type_text
    #         and y0 - last_y1 < 0.15 * h_im
    #         and (
    #             label.txt.startswith("Fig") or label.txt.startswith("FIG")
    #         )  # todo regex, skip non-alpha beginning
    #     )
    # ):
    #     if not last_entry:
    #         last_id = "X-Y"
    #     elif last_entry.label_type not in label_type_img:
    #         last_id = f"{pg_num}-Y"
    #     else:
    #         last_id = f"{last_entry.pgs[0]}-{last_entry.labels[0]}"

    #     label.label_type = "FigureCaption_" + last_id

    #     out_dir = Path(self.img_dir / f"{ch_i}")
    #     out_dir.mkdir(exist_ok=True)
    #     with open(out_dir / f"{last_id}.txt", "w") as f:
    #         f.write(label.txt)
    # TODO body-text figure association
