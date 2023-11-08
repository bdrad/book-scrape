import json
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Set, Tuple

import fire
import layoutparser as lp
import matplotlib.pyplot as plt
import pdfplumber
from layoutparser.elements import Interval, Layout, TextBlock
from pdfplumber.display import PageImage
from pdfplumber.page import Page
from PIL import Image
from tqdm import tqdm

from keppel.cleaning import clean_fonts, clean_fontstr, compact_json, round_to_nearest_k
from keppel.config import BookConfig
from keppel.tracker import ChapterTracker

DATA = Path("scrape")
assert DATA.is_dir()

from collections import Counter


def extract_fonts(pg: Page, k=4) -> List[Tuple[str, float]]:
    count = Counter()
    for ch in pg.chars:
        # out.add((ch['fontname'], ch['size']))
        name, size = ch["fontname"], ch["size"]
        size = round_to_nearest_k(size, k=k) if round else size
        count[(name, size)] += 1

    out = [(clean_fontstr(fname), fsize) for (fname, fsize), _cnt in count.most_common()]

    # remove duplicates, but maintain ordering
    dedup = []
    seen = set()
    for fname, fsize in out:
        if (fname, fsize) not in seen:
            seen.add((fname, fsize))
            dedup.append((fname, fsize))
    return dedup


class Parser(object):
    def __init__(
        self,
        fname: str,
        pad=0.010,
        # pad=0.0075,
        # pad=0.006,
        resolution=123,
        laparams=dict(detect_vertical=False),
        # laparams=dict(detect_vertical=False, word_margin=0.08)    # this doesn't cause any difference?
        **kwargs,
    ) -> None:
        assert 0 <= pad < 1

        self.fname = Path(fname)
        self.pad = pad
        self.resolution = resolution
        self.laparams = laparams

        self.pdf = pdfplumber.open(fname, laparams=laparams).pages

        self.outdir = Path("scrape_out")
        self.outdir /= self.fname.stem
        self.outdir.mkdir(exist_ok=True)

        self.cfg: BookConfig = BookConfig(fname)

        self.model = lp.models.Detectron2LayoutModel(
            # 'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config",
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.9],
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.77],
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.72],
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.7],
            # extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        )

        self.begin(**kwargs)

    # @property (??? idk what property actually does)
    def _get_labels(self, im, return_split_idx=False, pad=0.05) -> Layout:
        # def get_labels(pg, resolution=123) -> Layout:
        # im = pg.to_image(resolution=resolution).annotated
        if type(im) is PageImage:
            im = im.annotated

        w, h = im.size

        layout = self.model.detect(im)  # Detect the layout of the input image

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

        # canvas_height=h, canvas_width=w
        left_width = w / 2 * (1.0 + pad)
        left_interval: Interval = lp.Interval(0, left_width, axis="x").put_on_canvas(im)
        left_blocks = text_blocks.filter_by(left_interval, center=True)

        # conjugates
        right_blocks = [b for b in text_blocks if b not in left_blocks]

        sort_key = lambda b: b.coordinates[1]
        left_blocks = sorted(left_blocks, key=sort_key)
        right_blocks = sorted(right_blocks, key=sort_key)

        text_blocks = [b.set(id=idx) for idx, b in enumerate(left_blocks + right_blocks)]

        for i, b in enumerate(text_blocks):
            for c in text_blocks[:i] + text_blocks[i + 1 :]:  # get all but b
                if b.is_in(c, center=True):
                    b = b.set(parent=c.id)

                    # TODO LOG THIS
                    print(f"%%% Text block {b.id} is inside {b.parent} -- skipping")

        if return_split_idx:
            return text_blocks, len(left_blocks)
        else:
            return text_blocks

    def begin(self):
        chapters = self.cfg.chapter_range()
        for i, ch in enumerate(chapters, 1):
            print(f"Processing ch{i}")
            start, end = ch
            pages = self.pdf[start:end]  # todo make fctn
            tracker = ChapterTracker()

            progress = tqdm(pages)
            for pg in progress:
                pg_num = pg.page_number
                im = pg.to_image(resolution=self.resolution).annotated
                w_im, h_im = im.size
                w_pg, h_pg = pg.width, pg.height
                w_ratio, h_ratio = w_pg / w_im, h_pg / h_im

                labels = self._get_labels(im)
                for label_num, label in enumerate(labels):
                    # print(label)
                    label: TextBlock
                    if label.parent is not None:
                        print(f"%%% Text block {label.id} is inside {label.parent} -- skipping")
                        continue
                    pad_x, pad_y = self.pad * w_im, self.pad * h_im
                    label = label.pad(left=pad_x, right=pad_x, top=pad_y, bottom=pad_y)
                    x0, y0, x1, y1 = label.coordinates
                    x0, x1 = x0 * w_ratio, x1 * w_ratio
                    y0, y1 = y0 * h_ratio, y1 * h_ratio
                    # x0,x1 = (x0-pad*w_im)*w_ratio, (x1+pad*w_im)*w_ratio
                    # y0,y1 = (y0-pad*h_im)*h_ratio, (y1+pad*h_im)*h_ratio

                    kind = label.type
                    # assert kind in ("Text", "Title")

                    area = pg.within_bbox((x0, y0, x1, y1))
                    fonts = extract_fonts(area)
                    txt = area.extract_text()
                    tracker.add_txt(txt, kind, fonts, pg_num, label_num)

            tracker.close()

            with open(self.outdir / f"ch{i}_tracker.json", "w", encoding="utf8") as f:
                print(f"*** saving to {f.name}")
                data = tracker.get_entries()
                data = [
                    {k: (list(v) if type(v) is set else v) for k, v in entry.items()}
                    for entry in data
                ]  # json-ify: tuples -> lists

                # json.dump(data, f, indent=2, ensure_ascii=False)
                s = json.dumps(data, indent=2, ensure_ascii=False)
                s = compact_json(s)
                f.write(s)

        # figures first: get all figures and associated captions
        # store these figure caption fonts info to use as exclude list later


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(Parser)
    else:
        print("No arguments given, running test case")
        fname = Path("scrape/Chest - Webb - Fundamentals of Body CT (4e).pdf")
        parser = Parser(fname)

        pg = parser.pages[0]

        # TODO make this function of parser?
        im = pg.to_image(resolution=123).annotated
        sorted_boxes = parser._get_labels(im)
        lp.draw_box(im, sorted_boxes, box_width=3, show_element_id=True)
