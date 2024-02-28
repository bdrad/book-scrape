import json
import logging
import os
import random
import sys
from collections import Counter
from operator import itemgetter
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import cv2
import fire
import matplotlib.pyplot as plt
import numpy as np
import pdfplumber
from pdfplumber.display import PageImage
from pdfplumber.page import Page
from tqdm import tqdm

from keppel.cleaning import clean_fontstr
from keppel.config import BookConfig
from keppel.tracker import EarlyExitException, JoinTracker, Label, LabelTypes, Tracker
from keppel.utils import round_to_nearest_k

logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

DATA = Path("scrape")
assert DATA.is_dir()

FONT_KINDS = ("text", "head", "bad")


def extract_fonts(pg: Page, round_k=4, clean=False) -> List[Tuple[Tuple[str, float], int]]:
    count = Counter()
    for ch in pg.chars:
        name, size = ch["fontname"], ch["size"]
        if clean:
            name = clean_fontstr(name)
        size = round_to_nearest_k(size, k=round_k) if round_k else size
        entry = (name, size)
        count.update([entry])

    return [[[name, size], freq] for (name, size), freq in count.items()]


class Parser(object):
    def __init__(self, fname: str, start_ch=1) -> None:
        self.fname = Path(fname)
        assert self.fname.exists(), f"{fname} not found"

        self.start_ch = start_ch
        self.__pdf = None  # lazy load
        self.outdir = Path("scrape_out")
        self.outdir /= self.fname.stem
        self.rawdir = self.outdir / "raw"
        self.cleandir = self.outdir / "clean"
        self.img_dir = self.outdir / "imgs"
        for dir in (self.outdir, self.rawdir, self.cleandir, self.img_dir):
            dir.mkdir(exist_ok=True)

        self.cfg: BookConfig = BookConfig(fname)

        # https://layout-parser.readthedocs.io/en/latest/notes/modelzoo.html
        cfg_model: dict = self.cfg.data["detectron"]
        # self._model_name = str(cfg_model["model_name"])
        # assert "PubLayNet" in self._model_name
        # self._model_co = float(cfg_model["score_co"])
        # assert 0 <= self._model_co <= 1
        self.pad = float(cfg_model["box_pad"])
        assert 0 <= self.pad < 1
        self.resolution = int(cfg_model["resolution"])

        # self._label_map = {0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
        self.__model = None  # lazy load

    @property
    def pdf(self) -> List[Page]:
        if self.__pdf is None:
            laparams = dict(detect_vertical=False)
            # laparams=dict(detect_vertical=False, word_margin=0.08)    # this doesn't cause any difference?
            self.__pdf = pdfplumber.open(self.fname, laparams=laparams).pages
        return self.__pdf

    @property
    def model(self):
        if self.__model is None:
            from docxchain.pipelines.document_structurization import DocumentStructurization

            # print(f"Loading model {self._model_name} with score cutoff {self._model_co}")
            print("Loading DocXLayout model")
            self.__model = DocumentStructurization(
                dict(
                    layout_analysis_configs=dict(
                        from_modelscope_flag=False,
                        model_path="/home/DocXLayout_231012.pth",
                    ),
                    text_detection_configs=dict(
                        from_modelscope_flag=True,
                        model_path="damo/cv_resnet18_ocr-detection-line-level_damo",
                    ),
                    text_recognition_configs=dict(
                        from_modelscope_flag=True,
                        model_path="damo/cv_convnextTiny_ocr-recognition-document_damo",  # alternatives: 'damo/cv_convnextTiny_ocr-recognition-scene_damo', 'damo/cv_convnextTiny_ocr-recognition-general_damo', 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo'
                    ),
                    formula_recognition_configs=dict(
                        from_modelscope_flag=False,
                        image_resizer_path="/home/LaTeX-OCR_image_resizer.onnx",
                        encoder_path="/home/LaTeX-OCR_encoder.onnx",
                        decoder_path="/home/LaTeX-OCR_decoder.onnx",
                        tokenizer_json="/home/LaTeX-OCR_tokenizer.json",
                    ),
                )
            )
        return self.__model

    # TODO renew overlap-checking code for new model
    # TODO search for missing text
    # * if last label figure and current is text
    def _get_labels(
        self, im, page: Page, return_split_idx=False, model=None, layout=None
    ) -> List[Label]:
        if isinstance(im, PageImage):
            im = im.annotated.original
        elif isinstance(im, str):
            im = cv2.imread(im, cv2.IMREAD_COLOR)
        if type(im) is not np.ndarray:
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        assert type(im) is np.ndarray

        w_im, h_im = im.shape[1], im.shape[0]
        w_pg, h_pg = page.width, page.height

        layout = layout or (model or self.model)(im)
        # print(layout)

        out = []
        for category in layout:
            # idx, label_type, bbox = (category[s] for s in ("category_index", "category_name", "region_poly"))
            idx, label_type, bbox = itemgetter("category_index", "category_name", "region_poly")(
                category
            )
            bbox = np.array(bbox).reshape(-1, 2)
            x0, y0, x1, y1 = bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()
            assert x0 < x1 and y0 < y1
            bbox = (x0, y0, x1, y1)

            label = Label(page.page_number, idx, label_type, bbox)
            label.calc_bbox_pad(w_im, h_im, w_pg, h_pg)
            out.append(label)
            # print('===\n',idx, label_type, txt)
            # category['text_list']

        def _sort_labels(labels: List[Label], left_scale=0.98) -> List[Label]:
            assert 0 <= left_scale <= 1
            sort_key = lambda b: b.bbox_raw[1]  # sort by top-most y
            labels = sorted(labels, key=sort_key)

            left_width = left_scale * w_im / 2
            left_blocks = [l for l in labels if l.bbox_raw[0] < left_width]
            right_blocks = [r for r in labels if r not in left_blocks]  # conjugates
            blocks = left_blocks + right_blocks

            for i, b in enumerate(blocks):
                b.id = i
                # for c in blocks[:i] + blocks[i + 1 :]:  # get all but b
                #     if b.is_in(c, center=True):
                #         # later on, parent used as idicator to skip
                #         b = b.set(parent=c.id)

            if return_split_idx:
                return blocks, len(left_blocks)
            return blocks

        out = _sort_labels(out)
        return out

    def determine_co(self, co_base: float = None, co_delta: float = 0.05, co_n: int = 6):
        #      assert (
        #          self.cfg.use_pdfplumber is False
        #      ), "Cannot map fonts to categories with pdfplumber (requires detectron model)"
        #     possible_pages = range(self.cfg.chapters[0], self.cfg.chapters[-1])
        #     random.seed(0)
        #     pages = random.sample(possible_pages, 5)

        #     co_base = co_base or self.cfg["detectron"]["score_co"]
        #     cutoffs = [
        #         v
        #         for i in range(-co_n // 2 + 1, co_n // 2 + 1)
        #         if (0 <= (v := co_base + i * co_delta) <= 1)
        #     ]
        #     n = len(cutoffs)

        #     imgs = [self.pdf[i].to_image(resolution=self.resolution).annotated for i in pages]
        #     results = []

        #     for co in tqdm(cutoffs):
        #         tmp_model = lp.models.Detectron2LayoutModel(
        #             self._model_name,
        #             extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", co],
        #             label_map=self._label_map,
        #         )
        #         outs = []
        #         for im in imgs:
        #             sorted_boxes = parser._get_labels(im, model=tmp_model)
        #             o = lp.draw_box(im, sorted_boxes, box_width=3, show_element_id=True)
        #             outs.append(o)
        #         del tmp_model
        #         results.append(outs)

        #     # plt.switch_backend("TkAgg")
        #     w, h = imgs[0].size
        #     w, h = [4 * k / w for k in (w, h)]
        #     fig, axs = plt.subplots(len(pages), n, figsize=(n * w, len(pages) * h))
        #     fig.tight_layout(pad=0.0)
        #     # gs = gridspec.GridSpec(len(pages), n, figure=fig)
        #     # gs.update(wspace=0.025, hspace=0.025)
        #     for j, (co, outs) in enumerate(zip(cutoffs, results)):
        #         for i, o in enumerate(outs):
        #             if axs.ndim == 1:
        #                 coord = (i,)
        #             else:
        #                 coord = (i, j)
        #             axs[*coord].imshow(o)
        #             axs[*coord].axis("off")
        #             if i == 0:
        #                 axs[*coord].set_title(f"co={co:.3f}")
        #     # plt.show()
        #     plt.subplots_adjust(wspace=0, hspace=0)
        #     fig.savefig(self.outdir / "co.png", bbox_inches="tight", dpi=300, pad_inches=0)
        raise NotImplementedError

    def _iter_ch_pages(self) -> Iterator:
        """
        First yields the number of chapters, then yields the pages for each chapter
        """
        if self.cfg.chapters is None:
            pdfs = sorted(self.fname.glob("*.pdf"))
            pdfs = pdfs[self.start_ch - 1 :]
            yield len(pdfs)
            for pdf in pdfs:
                with pdfplumber.open(pdf) as p:
                    yield p.pages
        else:
            yield len(self.cfg.chapters)
            chapters = self.cfg.chapters[self.start_ch - 1 :]
            for start, end in chapters:
                yield self.pdf[start:end]

    def extract_raw(self, extract_figs=True):
        itr = self._iter_ch_pages()
        num_chapters = next(itr)
        for ch_i, pages in enumerate(itr, self.start_ch):
            tpages = tqdm(pages)
            tpages.set_postfix_str(f"ch{ch_i}/{num_chapters}")
            tracker = Tracker()
            for pg in tpages:
                pg_num = pg.page_number
                im = pg.to_image(resolution=self.resolution).annotated
                w_im, h_im = im.size
                w_pg, h_pg = pg.width, pg.height

                labels: List[Label] = self._get_labels(im, pg)
                for label in labels:
                    bb = label.calc_bbox_pad(w_im, h_im, w_pg, h_pg)
                    area = pg.within_bbox(bb, strict=False)

                    if label.label_type != LabelTypes.FIG:
                        label.fonts = extract_fonts(area)
                        label.txt = area.extract_text()
                    elif extract_figs and label.label_type in LabelTypes.S_IMG:
                        # pg_crop = pg.crop(bb, strict=False)
                        # img = pg_crop.to_image(resolution=self.resolution)
                        img = area.to_image(resolution=self.resolution)
                        out_dir = Path(self.img_dir / f"{ch_i}")
                        out_dir.mkdir(exist_ok=True)
                        img.save(out_dir / f"{pg_num}-{label.id}.png")

                    tracker.add_label(label)
                    tracker.to_file(self.rawdir / f"{ch_i}.json")

    # TODO FigureCaption support -- break up bad into table / figurecaption / figure
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
        for ch_i in tqdm(range(self.start_ch, len(self.cfg.chapters))):
            tracker = JoinTracker(self.cfg)
            with open(self.rawdir / f"{ch_i}.json", "r", encoding="utf8") as f:
                data = json.load(f)
                for entry in data:
                    pgs, label_type, txt, fonts, bbox_raw, bbox_pad, id = itemgetter(
                        "pgs", "label_type", "txt", "fonts", "bbox_raw", "bbox_pad", "id"
                    )(entry)
                    label = Label(pgs[0], id, label_type, bbox_raw, txt, fonts)
                    # label.bbox_pad = bbox_pad
                    try:
                        tracker.add_label(label)
                    except EarlyExitException:
                        break

            tracker.to_file(self.cleandir / f"{ch_i}.json")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fire.Fire(Parser)
    else:
        print("No arguments given, running test case")
        # fmt: off
        # ====
        fname = "Chest - Webb - Fundamentals of Body CT (4e).pdf"
        # fname = "Chest - Elicker - HRCT of the Lungs 2e.pdf"  # poorly scanned
        # fname = "US - Ultrasound Requisites (3e).pdf"
        # fname = "Breast - ACR - BIRADS_ATLAS (2013).pdf"
        # fname = "MSK - Greenspan - Orthopedic Imaging (6e).pdf"
        # fname = "MSK - Helms - Fundamentals of Skeletal Radiology (4e).pdf"
        # fname = "Neuro - Lane - The Temporal Bone Textbook.pdf"
        # fname = "NM - Mettler - Nuclear Medicine (6e).pdf"
        # fname = "Peds - Donnelly - Pediatric Imaging The Fundamentals.pdf"
        # === Directories
        # fname = "Arthritis in B&W 3e"
        # === Poor scans:
        # fname = "General - Mandell - Core Radiology (1e).pdf"   # poorly parsed
        # fname = "General - Weissleder - Primer of Diagnostic Imaging (5e).pdf"
        # === Buggy cases
        # fname = "General - Brant _ Helms - Fundamentals of Diagnostic Radiology (4e).pdf"  # !crashed
        # fname = "EM - Raby - Accident & Emergency Radiology (3e).pdf"  # simply doesn't load??
        # ===
        # fname = "test"
        # fname = "output.pdf"

        # for fname in ["Cardiac Imaging Requisites 4e",
        #     "Duke Review of MRI Principles",
        #     "Emergency Radiology Requisites 2e",
        #     "Fundamentals of Body CT 4e",
        #     "Gastrointestinal Requisites 4e",
        #     "Pediatric Imaging Fundamentals",
        #     "Ultrasound Requisites 3e",
        #     "Vascular and Interventional Radiology Requisites 2e"]:

        fname = Path("scrape/" + fname)
        print(str(fname))

        parser = Parser(fname, start_ch=1)

        # parser.determine_co(co_base=0.7)
        # parser.extract_raw()
        # parser.determine_fonts()
        parser.clean_raw()
