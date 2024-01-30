import re
from typing import List, Set, Tuple

import nltk

from keppel.utils import round_to_nearest_k

try:
    from nltk.corpus import wordnet31 as wordnet
except ImportError:
    nltk.download("wordnet31")
    from nltk.corpus import wordnet31 as wordnet

from keppel.config import Config

TOK_SENTENCE_TERMS = (".", "!", "?")
TOK_SENTENCE_CONTS = (",", ":", ";")
TERM_PUNC_STR = "".join(TOK_SENTENCE_TERMS)


cfg = Config()
data = cfg.data["base"]["cleaning"]  # TODO this should be determined by book config, not base

hyphen_extras = set(data["hyphen"]["extras"])
hyphen_prefixs = set(data["hyphen"]["prefixs"])
hyphen_suffixs = set(data["hyphen"]["suffixs"])

hyphen_extras, hyphen_prefixs, hyphen_suffixs = [
    set(map(lambda s: s.lower(), data)) for data in (hyphen_extras, hyphen_prefixs, hyphen_suffixs)
]

hyphen_corpus = set(filter(lambda w: "-" in w, wordnet.words()))
hyphen_corpus = hyphen_corpus.union(hyphen_extras)


def clean_fontstr(font: str):
    # FFIOJI+Univers-CondensedBold => Univers
    # UMTALE+JansonText-Roman => JansonText
    inner_font = r"^([A-Z]+)\+(\w+?)(?:-(\w+?))+$"
    return re.sub(inner_font, r"\2", font) if re.search(inner_font, font) else font


def clean_fonts(fonts: List[Tuple[str, float]]) -> Set[Tuple[str, float]]:
    if not fonts:
        return fonts
    return set([(clean_fontstr(f), round_to_nearest_k(s, k=4)) for f, s in fonts if f])


def dehyphenate_string(input_string, pg_num=None):
    out = ""
    dehyphens = []

    # remove surronding space from either left or right -- but don't conjoin
    if re.search(r"\w\s+-\s+\w", input_string):
        return input_string, dehyphens
    input_string = re.sub(r"\s*-\s*", r"-", input_string)

    re_number = r"^\d+(\.\d+)?-(.+)$"  # e.x. 15.25-mm
    re_word_extract = r"^(\W*)([\w’-]*)(\W*)$"
    re_hyph_split = r"^([^-]*)-(.*)$"

    # this will split, but preserve the whitespace
    words = re.split(r"(?<=[\s+])", input_string)
    for full_word in words:
        if not full_word:
            continue
        if "-" not in full_word or re.search(re_number, full_word):
            out += full_word
            continue

        try:
            match = re.search(re_word_extract, full_word)
            pre, word, post = match.groups()
            # print(pre,word,post)
            if "-" not in word:
                raise AttributeError

            match = re.search(re_hyph_split, word)
            word_pre, word_post = match.groups()
        except AttributeError:
            print(
                f"###{f' PG {pg_num} ::' if pg_num else ''} Failed to dehyphenate word: {full_word}"
            )
            out += full_word
            continue

        if word_pre.lower() not in hyphen_prefixs and word_post.lower() not in hyphen_suffixs:
            if word.lower() not in hyphen_corpus:
                # print('Dehyphenating word: ', word)
                dehyphens.append(word)
                word = word_pre + word_post
        out += pre + word + post
    return out, dehyphens


def process_text(text: str, log=False, pg_num=None, placeholder=None) -> str:
    placeholder = placeholder or ""

    text = re.sub(
        r"([" + TERM_PUNC_STR + "," + r"])([a-z][^\.] ?)", r"\1 \2", text, flags=re.IGNORECASE
    )  # e.x. abc.def -> abc. def; but not e.g. -> e. g.
    text = re.sub(
        r"figs?\.", "figure", text, flags=re.IGNORECASE
    )  # because we are splitting on '.', we need to replace 'Fig.' with 'figure'
    text = re.sub(
        r"(figure|table|image)s?\s+\d+\.", r"\1 ", text, flags=re.IGNORECASE
    )  # because we are splitting on '.', we need to replace 'figure 9.' with 'figure'
    text = re.sub(r"([\)\]\}])(\w)", r"\1 \2", text)  # e.x. (lorem)ipsum -> (lorem) ipsum
    text = re.sub(r"(\w)([\(\[\{])", r"\1 \2", text)  # e.x. lorem(ipsum) -> lorem (ipsum)
    text = re.sub(r"([a-z])([\d])", r"\1 \2", text, flags=re.IGNORECASE)  # e.x. abc1 -> abc 1
    text = re.sub(
        r"([a-ln-z])([A-Z])", r"\1 \2", text
    )  # e.x. abcDef -> abc Def; but not m* e.x. mA, mSv
    text = re.sub(r"([’']s)(\w)", r"\1 \2", text)  # e.x. 'sA -> 's A

    out = ""
    sentences = re.split(
        r"(?<=[" + TERM_PUNC_STR + r"] )", text
    )  # this will split, but preserve the delimiter
    # print(sentences)
    for sent in sentences:
        if not sent:
            continue

        # TODO implement placeholder; give more context to downstream processing
        ref_whole = r"([fF](igure|IGURE)|[tT](able|ABLE)|[iI](mage|MAGE))\s+[\dA-Z]"

        ref_parenthesis = (
            r" ?\(.*" + ref_whole + r".*?\)"
        )  # e.x. (lorem Table 1 ipsum) or (Fig. 2.1)
        sent = re.sub(ref_parenthesis, "", sent, flags=re.IGNORECASE)

        if re.search(ref_whole, sent):
            continue

        sent, dehyphs = dehyphenate_string(sent, pg_num=pg_num)
        if dehyphs and log:
            print(f"{'PG  :: d' + pg_num if pg_num else 'D'}e-hyphenated {dehyphs}")
        out += sent

    return out  # .strip()


if __name__ == "__main__":
    # fmt: off
    # assert (res:=dehyphenate_string((txt:="non-Hodgkin’s"))) == (txt,[]), res
    # assert (res:=dehyphenate_string((txt:="max-something"))) == (txt,[]), res
    # assert (res:=dehyphenate_string((txt:="max -something"))) == ("max-something",[]), res  # remove inner spaces
    # assert (res:=dehyphenate_string((txt:="max- something"))) == ("max-something",[]), res  # remove inner spaces
    # assert (res:=dehyphenate_string((txt:="max - something"))) == (txt,[]), res  # don't conjoin
    # assert (res:=dehyphenate_string((txt:="minimum-otherthing"))) == (txt,[]), res
    # assert (res:=dehyphenate_string((txt:="highest-otherthing"))) == (txt,[]), res
    # assert (res:=dehyphenate_string((txt:="False Positives: x-ray is a hyphenated word. 15.25-mm is a hyphenated number."))) == (txt,[]), res
    # assert (res:=dehyphenate_string( \
    #     (txt:="True Positive: This is an ex-ample of a de-hyphenated string"))) == \
    #     ("True Positive: This is an example of a dehyphenated string",['ex-ample', 'de-hyphenated']), res
    # assert (res:=dehyphenate_string((txt:=" Floats + keeps outer whitespace: Gantry rotation time is usually about 0.5 seconds. "))) == (txt,[]), res

    # assert (res:=process_text((txt:="Gantry rotation time is usually about 0.5 seconds"))) == txt, res
    # assert (res:=process_text((txt:="The formula relating scan parameters for MDCT is shown in Fig. 1-1."))) == '', res
    # assert (res:=process_text((txt:="However, several general principles apply to all chest scans (TABLE 1-1)."))) == 'However, several general principles apply to all chest scans.', res
    # assert (res:=process_text((txt:="Non-Hodgkin’s"))) == txt, res
    # assert (res:=process_text((txt:="max-something"))) == txt, res
    # assert (res:=process_text((txt:="minimum-otherthing"))) == txt, res
    # assert (res:=process_text((txt:="highest-otherthing"))) == txt, res
    # assert (res:=process_text((txt:="e.g."))) == txt, res
    # assert (res:=process_text((txt:=".1"))) == txt, res

    # assert (res:=process_text((txt:="lorem)ipsum"))) == "lorem) ipsum", res
    # assert (res:=process_text((txt:="abc1"))) == "abc 1", res
    # assert (res:=process_text((txt:=".a "))) == ". a ", res
    # assert (res:=process_text((txt:=".a"))) == ".a", res
    # assert (res:=process_text((txt:="abcDef"))) == "abc Def", res
    # assert (res:=process_text((txt:="lorem(ipsum)"))) == "lorem (ipsum)", res
    # assert (res:=process_text((txt:="quickly,excellent"))) == "quickly, excellent", res

    # assert (res:=process_text((txt:="figs."))) == "figure", res
    # assert (res:=process_text((txt:="fig."))) == "figure", res
    # assert (res:=process_text((txt:="(fig. A)"))) == "", res
    # assert (res:=process_text((txt:="(fig. 3 description)"))) == "", res
    # assert (res:=process_text((txt:="(figure\n9. SAE)"))) == "", res
    # assert (res:=process_text((txt:="pre (fig. 1)."))) == "pre.", res
    # assert (res:=process_text((txt:="Slice Thickness and Pitch (Table Excursion)"))) == txt, res
    # fmt: on

    print("All tests passed.")

    fonts = [
        ["UMTALE+JansonText-Roman", 10.0],
        ["PSFMKZ+JansonText-Italic", 10.0],
        ["FFIOJI+Univers-CondensedBold", 10.0],
        ["UMTALE+JansonText-Roman", 12.0],
        ["FUCJOE+JansonText-Roman-SC800", 27.0],
    ]
    print(clean_fonts(fonts))

    assert round_to_nearest_k(1.3, 4) == 1.25
    assert round_to_nearest_k(1.2, 4) == 1.25
