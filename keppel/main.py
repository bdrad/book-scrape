import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import fire
import layoutparser as lp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pdfplumber
from layoutparser.elements import Interval, Layout, TextBlock
from pdfplumber.display import PageImage
from pdfplumber.page import Page
from tqdm import tqdm

from keppel.cleaning import clean_fontstr
from keppel.config import BookConfig
from keppel.tracker import EarlyExitException, JoinTracker, PreTracker
from keppel.utils import round_to_nearest_k

DATA = Path("scrape")
assert DATA.is_dir()


FONT_KINDS = ("text", "head", "bad")


def extract_fonts(pg: Page, round_k=4) -> list[list[str, float], int]:
    count = Counter()
    for ch in pg.chars:
        name, size = ch["fontname"], ch["size"]
        name = clean_fontstr(name)
        size = round_to_nearest_k(size, k=round_k) if round_k else size
        entry = (name, size)
        count.update([entry])

    return [[[name, size], freq] for (name, size), freq in count.items()]
    # out = [(name, size, freq) for (name, size), freq in count.items()]
    # return out
    # out = list(count.items()) # drop counts

    # # remove duplicates, but maintain ordering
    # dedup = []
    # seen = set()
    # for entry in out:
    #     if entry not in seen:
    #         seen.add(entry)
    #         dedup.append(entry)
    # return dedup


class Parser(object):
    def __init__(self, fname: str) -> None:
        self.fname = Path(fname)
        assert self.fname.is_file(), f"File {fname} not found"

        laparams = dict(detect_vertical=False)
        # laparams=dict(detect_vertical=False, word_margin=0.08)    # this doesn't cause any difference?
        self.pdf = pdfplumber.open(fname, laparams=laparams).pages

        self.outdir = Path("scrape_out")
        self.outdir /= self.fname.stem
        self.rawdir = self.outdir / "raw"
        self.cleandir = self.outdir / "clean"
        for dir in (self.outdir, self.rawdir, self.cleandir):
            dir.mkdir(exist_ok=True)
        for dir in (self.rawdir, self.cleandir):
            img_dir = dir / "imgs"
            img_dir.mkdir(exist_ok=True)

        self.cfg: BookConfig = BookConfig(fname)
        cfg_model = self.cfg.data["detectron"]

        self.pad = cfg_model["box_pad"]
        assert 0 <= self.pad < 1
        # pad=0.010,
        # pad=0.0075,
        # pad=0.006,

        self.resolution = cfg_model["resolution"]

        # https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html
        self._model_name, _model_co = str(cfg_model["model_name"]), float(cfg_model["score_co"])
        assert "PubLayNet" in self._model_name
        self._label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        assert 0 <= _model_co <= 1
        print(f"Loading model {self._model_name} with score cutoff {_model_co}")

        self.model = lp.models.Detectron2LayoutModel(
            # 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            # "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
            self._model_name,
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.9],
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.77],
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.72],
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", _model_co],
            label_map=self._label_map,
        )

    # TODO ensure that the ordering here is correct
    # * notably, Webb Ch1 Pg1 has text prior to title (wasn't always like this, I thought..)
    # * could possible use overlapping text to determine ordering

    # TODO search for missing text
    # * if last label figure and current is text

    def _get_labels(self, im, return_split_idx=False, pad=0.05, model=None) -> List[Layout]:
        if type(im) is PageImage:
            im = im.annotated

        w, h = im.size

        layout = (model or self.model).detect(im)  # Detect the layout of the input image
        text_blocks: Layout = lp.Layout([b for b in layout if b.type in ("Text", "Title")])
        nontext_blocks: Layout = lp.Layout([b for b in layout if b.type in ("Figure", "Table")])

        # As there could be text region detected inside the figure region, we just drop them:
        text_blocks: Layout = lp.Layout(
            [
                b
                for b in text_blocks
                if not any(b.is_in(b_fig, center=True) for b_fig in nontext_blocks)
            ]
        )
        blocks = text_blocks + nontext_blocks

        # canvas_height=h, canvas_width=w
        left_width = w / 2 * (1.0 + pad)
        left_interval: Interval = lp.Interval(0, left_width, axis="x").put_on_canvas(im)
        left_blocks = blocks.filter_by(left_interval, center=True)

        # conjugates
        right_blocks = [b for b in blocks if b not in left_blocks]

        sort_key = lambda b: b.coordinates[1]
        left_blocks = sorted(left_blocks, key=sort_key)
        right_blocks = sorted(right_blocks, key=sort_key)

        blocks = [b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)]

        for i, b in enumerate(blocks):
            for c in blocks[:i] + blocks[i + 1 :]:  # get all but b
                if b.is_in(c, center=True):
                    # later on, parent used as idicator to skip
                    b = b.set(parent=c.id)

        if return_split_idx:
            return blocks, len(left_blocks)
        return blocks

    def determine_co(self, co_base: float = None, co_delta: float = 0.02, co_n: int = 5):
        pages = range(44, 48)  # todo get random pages (use seed)

        co_base = co_base or self.cfg["detectron"]["score_co"]
        cutoffs = [
            v for i in range(-co_n // 2, co_n // 2) if (0 <= (v := co_base + i * co_delta) <= 1)
        ]
        n = len(cutoffs)

        imgs = [self.pdf[i].to_image(resolution=self.resolution).annotated for i in pages]
        results = []

        for co in tqdm(cutoffs):
            tmp_model = lp.models.Detectron2LayoutModel(
                self._model_name,
                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", co],
                label_map=self._label_map,
            )
            outs = []
            for im in imgs:
                sorted_boxes = parser._get_labels(im, model=tmp_model)
                o = lp.draw_box(im, sorted_boxes, box_width=3, show_element_id=True)
                outs.append(o)
            del tmp_model
            results.append(outs)

        # plt.switch_backend("TkAgg")
        w, h = imgs[0].size
        w, h = [4 * k / w for k in (w, h)]
        fig, axs = plt.subplots(len(pages), n, figsize=(n * w, len(pages) * h))
        fig.tight_layout(pad=0.0)
        # gs = gridspec.GridSpec(len(pages), n, figure=fig)
        # gs.update(wspace=0.025, hspace=0.025)
        for i, o in enumerate(outs):
            for j, (co, outs) in enumerate(zip(cutoffs, results)):
                axs[i, j].imshow(o)
                axs[i, j].axis("off")
                if i == 0:
                    axs[i, j].set_title(f"co={co:.3f}")
        # plt.show()
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.savefig(self.outdir / "co.png", bbox_inches="tight", dpi=300, pad_inches=0)

    def extract_raw(self, extract_figs=True):
        chapters = self.cfg.chapter_range()
        for ch_i, ch in enumerate(chapters, 1):
            print(f"Processing ch{ch_i}")
            start, end = ch
            pages: List[Page] = self.pdf[start:end]  # todo make fctn
            pretracker = PreTracker()

            for pg in tqdm(pages):
                pg_num = pg.page_number
                im = pg.to_image(resolution=self.resolution).annotated
                w_im, h_im = im.size
                w_pg, h_pg = pg.width, pg.height
                w_ratio, h_ratio = w_pg / w_im, h_pg / h_im

                labels = self._get_labels(im)
                last_entry: TextBlock = None
                last_y1: float = 0.0
                for label_num, label in enumerate(labels):
                    # print(label)
                    label: TextBlock
                    kind = label.type

                    if label.parent is not None:
                        print(f"%%% Text block {label.id} is inside {label.parent} -- skipping")
                        continue

                    pad_x, pad_y = self.pad * w_im, self.pad * h_im
                    label = label.pad(left=pad_x, right=pad_x, top=pad_y, bottom=pad_y)
                    x0, y0, x1, y1 = label.coordinates  # left, top, right, bottom
                    x0, x1 = x0 * w_ratio, x1 * w_ratio
                    y0, y1 = y0 * h_ratio, y1 * h_ratio

                    if extract_figs and kind == "Figure":
                        pg_crop = pg.crop((x0, y0, x1, y1), strict=False)
                        img = pg_crop.to_image(resolution=self.resolution)
                        out_dir = Path(self.rawdir / f"imgs/{ch_i}")
                        out_dir.mkdir(exist_ok=True)
                        img.save(out_dir / f"{pg_num}-{label_num}.png")

                    area = pg.within_bbox((x0, y0, x1, y1), strict=False)
                    fonts = extract_fonts(area)
                    txt = area.extract_text()

                    if (
                        last_entry
                        and last_entry.label_type == "Figure"
                        and kind == "Text"
                        and y0 - last_y1 < 0.15 * h_im
                        and (txt.startswith("Fig") or txt.startswith("FIG"))
                    ):
                        last_id = f"{last_entry.pgs[0]}-{last_entry.labels[0]}"
                        kind = "FigureCaption-" + last_id
                        if extract_figs:
                            out_dir = Path(self.rawdir / f"imgs/{ch_i}")
                            out_dir.mkdir(exist_ok=True)
                            with open(out_dir / f"{last_id}.txt", "w") as f:
                                f.write(txt)

                    last_entry = pretracker.add_entry(pg_num, kind, label_num, txt, fonts)
                    last_y1 = y1

            pretracker.to_file(self.rawdir / f"{ch_i}.json")

    def determine_fonts(self, display=True, cutoff=0.10):
        assert 0 <= cutoff <= 1

        if self.cfg.get_fonts() and "y" != input(
            "Book already has fonts stored in the config file. Overwrite? [y/n]"
        ):
            return

        text_cnt = Counter()
        head_cnt = Counter()
        bad_cnt = Counter()

        for i in tqdm(range(1, len(self.cfg.chapters))):
            with open(self.rawdir / f"{i}.json", "r", encoding="utf8") as f:
                data = json.load(f)

            for entry in data:
                kind = entry["label_type"]
                fonts = entry["fonts"]
                fonts = {(name, size): freq for [name, size], freq in fonts}
                if kind == "Text":
                    text_cnt.update(fonts)
                elif kind == "Title":
                    head_cnt.update(fonts)
                else:
                    bad_cnt.update(fonts)

        if display:
            for kind, cnt in zip(FONT_KINDS, (text_cnt, head_cnt, bad_cnt)):
                print("=" * 50)
                print(f"# {kind.title()} fonts")
                for font, freq in cnt.most_common(6):
                    s_font = str(font).ljust(30)
                    s_freq = str(freq).rjust(6)
                    print(f"{s_font}: {s_freq}  ({freq / cnt.total():.2%})")
            print("=" * 50)

        text_fonts = []
        blacklist = set()
        for i, (font, cnt) in enumerate(text_cnt.most_common()):
            freq = cnt / text_cnt.total()
            if freq < cutoff:
                break

            top_h_font, top_h_cnt = head_cnt.most_common()[0]
            if font == top_h_font:
                blacklist.add((1, i))
                continue
            if font == bad_cnt.most_common()[0][0]:
                blacklist.add((2, i))
                continue

            text_fonts.append(font)

        fonts = [text_fonts, (head_fonts := []), (bad_fonts := [])]
        cnts = [text_cnt, head_cnt, bad_cnt]

        q = [(1, 0), (2, 0)]  # (kind, idx)
        q = list(filter(lambda x: x not in blacklist, q))
        while q:
            kind, idx = q.pop(0)
            font, cnt = cnts[kind].most_common()[idx]
            freq = cnt / cnts[kind].total()
            # print(font, cnt, freq)
            if freq < cutoff:
                continue
            other_idxs = [i for i in range(3) if i != kind]
            if all([font not in fonts[i] for i in other_idxs]):
                fonts[kind].append(font)
            if (nxt := (kind, idx + 1)) not in blacklist:
                q.append(nxt)

        self.cfg.write_fonts(fonts)

    def clean_raw(self):
        for i in tqdm(range(1, len(self.cfg.chapters))):  # glob instead?
            with open(self.rawdir / f"{i}.json", "r", encoding="utf8") as f:
                data = json.load(f)

            tracker = JoinTracker()

            for entry in data:
                txt, pg_num, label_type, label_num, fonts = (
                    entry["txt"],
                    entry["pgs"][0],
                    entry["label_type"],
                    entry["labels"][0],
                    entry["fonts"],
                )
                try:
                    tracker.add_entry(pg_num, label_type, label_num, txt, fonts)
                except EarlyExitException as e:
                    pass
            tracker.to_file(self.cleandir / f"{i}.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(Parser)
    else:
        print("No arguments given, running test case")
        fname = "Chest - Webb - Fundamentals of Body CT (4e).pdf"
        #
        # fname = "Chest - Felson - Principles of Chest Roentgenology (4e).pdf"
        # fname = "Chest - Elicker - HRCT of the Lungs 2e.pdf"
        # fname = "General - Brant _ Helms - Fundamentals of Diagnostic Radiology (4e).pdf"  # !crashed
        # fname = "General - Mandell - Core Radiology (1e).pdf"
        # fname = "General - Weissleder - Primer of Diagnostic Imaging (5e).pdf"
        fname = Path("scrape/" + fname)
        print(str(fname))

        parser = Parser(fname)

        # parser.determine_co(co_base=0.7, co_delta=0.02, co_n=5)
        parser.extract_raw()
        # parser.determine_fonts()
        # parser.clean_raw()

        # pg = parser.pdf[0]

        # # TODO make this function of parser?
        # im = pg.to_image(resolution=123).annotated
        # sorted_boxes = parser._get_labels(im)
        # lp.draw_box(im, sorted_boxes, box_width=3, show_element_id=True)
