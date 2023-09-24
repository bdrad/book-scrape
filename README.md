# book-scrape


Motivated by _[Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)_.


## todos

- [x] Use dependency manager; generate `requirements.txt`
- [ ] Better logging
- [ ] 


### Scraping
- [ ] Parse math equations, e.x. page 288 of _Chest - Webb - Fundamentals of Body CT (4e)_
- [ ] Parse tables? This may be difficult, and perhaps irrelevant for the purpose of training LLMs


### Post-processing
- [ ] Remove references to figures, tables, and other information that is not scrapped
- [ ] De-hyphenate words ([`TEXT_DEHYPHENATE` flag doesn't work](https://pymupdf.readthedocs.io/en/latest/app1.html#text-extraction-flags-defaults)?)
  - Greedy approach may not be sufficient. E.g. 'x-ray' is common


### books (to) support
- [ ] Chest - Elicker - HRCT of the Lungs 2e
- [ ] Chest - Felson - Principles of Chest Roentgenology (4e)
- [ ] Cardiac Imaging Requisites 4e
- [ ] Chest - Elicker - HRCT of the Lungs 2e
- [ ] Chest - Felson - Principles of Chest Roentgenology (4e)
- [x] Chest - Webb - Fundamentals of Body CT (4e)
- [ ] Emergency Radiology Requisites 2e
- [ ] Fundamentals of Body CT 4e
- [ ] General - Brant _ Helms - Fundamentals of Diagnostic Radiology (4e)
- [ ] General - Mandell - Core Radiology (1e)
- [ ] General - Weissleder - Primer of Diagnostic Imaging (5e)
- [ ] Vascular and Interventional Radiology Requisites 2e
