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
from keppel.tracker import EarlyExitException, JoinTracker, Label, LabelTypes, Tracker, FigTracker
from keppel.utils import round_to_nearest_k

logging.disable(logging.INFO)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # modelhub torch logs

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
    def __init__(self, fname: str, start_ch=1, device=0, outdir: Path = None) -> None:
        self.fname = Path(fname)
        self.__pdf = None  # lazy load
        assert self.fname.exists(), f"{fname} not found"
        self.start_ch = start_ch
        self.device = device

        # TODO change out base to shared drive
        self.outdir = outdir or Path("scrape_out")
        self.outdir /= self.fname.stem
        self.rawdir = self.outdir / "raw"
        self.cleandir = self.outdir / "clean"
        self.img_dir = self.outdir / "imgs"
        for dir in (self.outdir, self.rawdir, self.cleandir, self.img_dir):
            dir.mkdir(exist_ok=True)

        self.cfg: BookConfig = BookConfig(fname)
        self.resolution = int(self.cfg.data["im_resolution"])

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
            print("Loading DocXLayout model")
            from docxchain.pipelines.document_structurization import (
                DocumentStructurization,
            )

            model_cfg = dict(
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
                    model_path="damo/cv_convnextTiny_ocr-recognition-document_damo",
                    # 'damo/cv_convnextTiny_ocr-recognition-scene_damo',
                    # 'damo/cv_convnextTiny_ocr-recognition-general_damo',
                    # 'damo/cv_convnextTiny_ocr-recognition-handwritten_damo'
                ),
                formula_recognition_configs=dict(
                    from_modelscope_flag=False,
                    image_resizer_path="/home/LaTeX-OCR_image_resizer.onnx",
                    encoder_path="/home/LaTeX-OCR_encoder.onnx",
                    decoder_path="/home/LaTeX-OCR_decoder.onnx",
                    tokenizer_json="/home/LaTeX-OCR_tokenizer.json",
                ),
            )

            self.__model = DocumentStructurization(model_cfg, device=self.device)
        return self.__model

    def eval_model(self, im, model=None):
        if isinstance(im, PageImage):
            im = im.annotated.original
        elif isinstance(im, str):
            im = cv2.imread(im, cv2.IMREAD_COLOR)
        if type(im) is not np.ndarray:
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        assert type(im) is np.ndarray
        return (model or self.model)(im)

    # TODO search for missing text
    # * if last label figure and current is text
    def _get_labels(
        self, im, page: Page, return_split_idx=False, model=None, layout=None
    ) -> List[Label]:
        w_im, h_im = im.shape[1], im.shape[0]
        w_pg, h_pg = page.width, page.height

        layout = layout or self.eval_model(im, model)
        # print(layout)

        out = []
        for category in layout:
            # idx, label_type, bbox = (category[s] for s in ("category_index", "category_name", "region_poly"))
            idx, label_type, bbox = itemgetter("category_index", "category_name", "region_poly")(
                category
            )
            bbox = np.array(bbox).reshape(-1, 2)
            x0, y0, x1, y1 = (
                bbox[:, 0].min(),
                bbox[:, 1].min(),
                bbox[:, 0].max(),
                bbox[:, 1].max(),
            )
            assert x0 < x1 and y0 < y1
            bbox = (x0, y0, x1, y1)

            label = Label(page.page_number, idx, label_type, bbox)
            label.calc_bbox_pad(w_im, h_im, w_pg, h_pg)
            out.append(label)
            # print('===\n',idx, label_type, txt)
            # category['text_list']

        def _prune_overlap_labels(labels: List[Label]) -> List[Label]:
            return [b for b in labels if not any(b.is_in(c) for c in labels)]

        out = _prune_overlap_labels(out)

        def _sort_labels(labels: List[Label], left_scale=0.98) -> List[Label]:
            assert 0 <= left_scale <= 1
            sort_key = lambda b: b.bbox_raw[1]  # sort by top-most y
            labels = sorted(labels, key=sort_key)

            left_width = left_scale * w_im / 2
            left_blocks = [l for l in labels if l.bbox_raw[0] < left_width]
            right_blocks = [r for r in labels if r not in left_blocks]  # conjugates
            blocks = left_blocks + right_blocks

            if return_split_idx:
                return blocks, len(left_blocks)
            return blocks

        out = _sort_labels(out)

        for i, b in enumerate(out):
            b.id = i

        return out

    def visualize(self, n: int = 6):
        """
        Visualize the layout parsing of n sampled pages
        """
        assert (
            self.cfg.use_pdfplumber is False
        ), "Cannot map fonts to categories with pdfplumber (requires detectron model)"
        from docxchain.utilities.visualization import document_structurization_visualization

        possible_pages = range(self.cfg.chapters[0][0], self.cfg.chapters[-1][-1])
        random.seed(0)
        pages = random.sample(possible_pages, n)

        model = self.model
        for p in tqdm(pages):
            page = self.pdf[p]
            im = page.to_image(resolution=self.resolution).annotated
            im = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
            res = self.eval_model(im, model)
            output_image = document_structurization_visualization(res, im)
            cv2.imwrite(str(self.outdir / f"layout_{p}.png"), output_image)

    # TODO if just some of the chapters are done, don't re-do them
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
            jtracker = JoinTracker(self.cfg)
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
                    jtracker.add_label(label)
            tracker.to_file(self.rawdir / f"{ch_i}.json")
            jtracker.to_file(self.cleandir / f"{ch_i}.json")

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
            jtracker = JoinTracker(self.cfg, path=self.rawdir / f"{ch_i}.json")
            ...  # TODO
            jtracker.to_file(self.cleandir / f"{ch_i}.json")

    def figure_raw(self):
        for ch_i in tqdm(range(self.start_ch, len(self.cfg.chapters))):
            FigTracker(self.cfg, self.rawdir / f"{ch_i}.json", process=True)


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
        # fname = "Cardiac Imaging Requisites 4e"
        # fname = "Duke Review of MRI Principles"
        # fname = "Emergency Radiology Requisites 2e"
        # fname = "Fundamentals of Body CT 4e".pdf
        # fname = "Gastrointestinal Requisites 4e"
        # fname = "Pediatric Imaging Fundamentals"
        # fname = "Ultrasound Requisites 3e"
        # fname = "Vascular and Interventional Radiology Requisites 2e"
        # === Poor scans:
        # fname = "General - Mandell - Core Radiology (1e).pdf"   # poorly parsed
        # fname = "General - Weissleder - Primer of Diagnostic Imaging (5e).pdf"
        # === Buggy cases
        # fname = "General - Brant & Helms - Fundamentals of Diagnostic Radiology (4e).pdf"  # !crashed
        # fname = "EM - Raby - Accident & Emergency Radiology (3e).pdf"  # simply doesn't load??
        # ===
        # fname = "test"
        # fname = "output.pdf"
        # ===
        # fname = Path("scrape/" + fname)
        # print(str(fname))

        # parser = Parser(fname, start_ch=1)
        # # parser.determine_co(co_base=0.7)
        # # parser.extract_raw()
        # # parser.determine_fonts()
        # parser.clean_raw()

        for fname in [
            "Chest - Webb - Fundamentals of Body CT (4e).pdf",
            # "Chest - Elicker - HRCT of the Lungs 2e.pdf",
            # "US - Ultrasound Requisites (3e).pdf",

            # "General - Brant & Helms - Fundamentals of Diagnostic Radiology (4e).pdf", # start ch12
            # "Breast - ACR - BIRADS_ATLAS (2013).pdf",
            # "MSK - Greenspan - Orthopedic Imaging (6e).pdf",
            ]:
            parser = Parser(
                fname="scrape/"+fname,
                start_ch=1,
                device=1,
                outdir=Path("./scrape_out")
                # outdir=Path("/mnt/sohn2022/Vogel/scrape_out")
            )
            parser.visualize(n=3)
            # parser.figure_raw()
            # parser.clean_raw()
