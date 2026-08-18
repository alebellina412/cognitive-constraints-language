# Data sources

This repository redistributes **no corpus text**. Everything under
`data_reduced/` is numeric: frequency vectors, frequency-of-frequency spectra,
network degree and strength, vocabulary-growth curves. The one exception is
`spgc_<lang>_phrases.json`, which holds the occurrence counts of the annotated
concepts and therefore contains those specific strings.

This document says exactly where each corpus comes from, which version was used,
how it was obtained, and under what licence. `notebooks/01_download_raw_data.ipynb`
automates what can be automated and points back here for the rest.

Three corpora are used:

| corpus | used for | size | acquisition |
| --- | --- | --- | --- |
| SPGC | everything except Figure S4 and Table S9 | ~8 GB | scripted |
| COREFL | Figure S4 (learners vs natives) | ~4 MB | manual download, scripted import and verification |
| PARSEME 1.1 | Table S9 (annotated multi-word expressions) | ~97 MB | manual download |

Figure S3 is SPGC too: it places the annotated concepts of §4 below on the same
phrase spectrum Figure 1B plots. Only Figure S4 and Table S9 use another corpus.

Two notebooks read `data_raw/`: `02_build_reduced_data.ipynb`, which reduces the
corpora, and `03_corpus_controls.ipynb`, which resamples the raw token streams
for Table S3 and the length controls. After those two have run, nothing reads it
again and it can be deleted.

---

## 1. SPGC — Standardized Project Gutenberg Corpus

The corpus behind every main-text figure.

**Version used: `SPGC-2018-07-18`** (55,905 books, ~3×10⁹ tokens). Pin this
release: Project Gutenberg itself changes over time, and the point of SPGC is
that a specific release does not.

### Where it comes from

- **Data** — Zenodo record [10.5281/zenodo.2422561](https://doi.org/10.5281/zenodo.2422561),
  version `SPGC-2018-07-18`.
- **Paper** — M. Gerlach & F. Font-Clos, *A Standardized Project Gutenberg Corpus
  for Statistical Analysis of Natural Language and Quantitative Linguistics*,
  Entropy 22(1):126 (2020), [doi:10.3390/e22010126](https://doi.org/10.3390/e22010126),
  arXiv:1812.08092.
- **Pipeline** — <https://github.com/pgcorpus/gutenberg>
- **Underlying source** — Project Gutenberg, <https://www.gutenberg.org>

### How to download it

```bash
python src/download_bulk.py                       # all three files
python src/download_bulk.py --only metadata counts
python src/download_bulk.py --only tokens
python src/download_bulk.py --check               # verify only, download nothing
```

Three files land in `data_raw/`:

| file | size | contents |
| --- | --- | --- |
| `SPGC-metadata-2018-07-18.csv` | ~10 MB | per-book language, author, title |
| `SPGC-counts-2018-07-18.zip` | ~1.5 GB | per-book 1-gram counts |
| `SPGC-tokens-2018-07-18.zip` | ~6.4 GB | per-book ordered token streams |

The download is resumable (`curl -L -C -`), idempotent — it skips files already
present and valid — and integrity-checked with `zipfile.testzip`. If you already
hold the SPGC bulk, put or symlink it into `data_raw/` under these names and run
`--check` instead.

`--check` also compares the MD5 of each file with
`manifests/spgc_checksums.json`, which records the three published for Zenodo
record 2422561. Size and `testzip` only say that a file is a complete zip; the
MD5 is what says it is *this* release. SPGC is the corpus behind every
main-text figure, so it is pinned the same way COREFL and PARSEME are.

The ordered token streams are needed because n-gram and network statistics depend
on word **order**, which the per-book counts discard.

### Which books, exactly

Single-language books only. Four of the five languages use every such book in the
release; English is subsampled, because using all of it would put English an order
of magnitude above the others in size.

**The English rule**, stated so it can be re-executed: shuffle the single-language
English ids with `numpy.random.default_rng(seed=0)`, take them in order until the
aggregated vocabulary reaches 5.5×10⁵ types. That gives 2,701 books.

The resulting book lists are committed and are version-independent:

- `manifests/corpus_manifest.json` — full per-language id lists under
  `languages.<code>.pg_ids`
- `manifests/english_subset_pgids.txt` — the English subsample

A few manifest books are absent from the frozen zips (a handful per language).
The reduction records this as `n_books_missing`, and `src/corpus_table.py` reports
it, so the corpus table always describes the data actually analysed.

### Licence

Project Gutenberg texts are in the public domain in the United States; check the
header of an individual work for its status in your country. The SPGC derived
files are distributed under the terms given in the Zenodo record.

---

## 2. COREFL — Corpus of English as a Foreign Language

Needed for Figure S4 only: English produced by learners, against native English.

**Version used: v2.0, released October 2025.** The copy this paper was computed
on was **downloaded on 26 July 2026**, and its exact file list with a sha256 per
file is `manifests/corefl_manifest.json`, whose `download` field records both
dates. The two are worth keeping apart: v2.0 added L1s that v1 did not have
(L1 Chinese, for instance), so "download COREFL" alone does not identify the
data — but neither does the release alone, since the site is served live and a
later download of the same release is not guaranteed to be the same bytes. The
manifest is what actually pins it; `--check` is what proves it.

### Where it comes from

- **Site** — <http://corefl.learnercorpora.com>
- **Cite** — Lozano, C., Díaz-Negrillo, A. & Callies, M. (2020), *Designing and
  compiling a learner corpus of written and spoken narratives: COREFL*, in
  *What's in a Narrative?*, 21–46, Peter Lang.
- **Licence** — CC BY-NC-ND 3.0 ES.

### Why this one cannot be scripted

COREFL is served by a web interface only. There is no bulk-download URL and no
API: `/search` and `/search_simple` return the same JavaScript shell, with no
form action and no archive endpoint. The download below is the only manual step
in the whole pipeline; everything after it is scripted.

### How to download it

Repeat **once per subcorpus** — once for the learners, once for the native
controls:

1. open <http://corefl.learnercorpora.com/search_simple>
2. **Subcorpus** → `Learners of L2 English`; **L1** → `L1 any`
3. leave **Words (optional)** *empty* — an empty query returns the whole
   subcorpus — and press **Search**
4. press **Download** at the bottom right of the result list
5. in the dialog: purpose → *For research*; **format → `Texts only`**
6. repeat from step 2 with **Subcorpus** set to the native-speaker controls

**Choose `Texts only`, not `Texts with metadata`.** The metadata format prepends
a header to each file which the tokeniser would count as running text, and the
CSV exports need a different reader. Nothing is lost: the file names carry the
metadata the pipeline uses.

Extract every archive into one folder — the importer classifies by file name, so
the folder layout does not matter — and check the count against the
"Results 1 to 50 of N" line the site displayed:

```bash
mkdir -p ~/Downloads/corefl && cd ~/Downloads/corefl
unzip ~/Downloads/<learners>.zip && unzip ~/Downloads/<natives>.zip
find ~/Downloads/corefl -name '*.txt' | wc -l
```

### Import and verify

Two steps, in this order. `--check` alone is not enough on a fresh clone: it
verifies `data_raw/corefl/`, which does not exist until the import has created
it.

```bash
python src/import_corefl.py --from-dir ~/Downloads/corefl   # classify into data_raw/corefl/
python src/import_corefl.py --check                         # verify against the committed manifest
```

`manifests/corefl_manifest.json` records a sha256 per file plus a note describing
the copy the paper used. If `--check` passes, you have the same data and will get
the same numbers; if it does not, the coverage table the importer prints tells you
how your copy differs.

To pin a deliberately different copy:

```bash
python src/import_corefl.py --from-dir ~/Downloads/corefl --write-manifest \
    --note 'v2.0 Oct 2025, subcorpora: learners + natives, Texts only, downloaded 2026-07-26'
```

Record the **download date** in the note as well as the release, as the line
above does: it is the only way to tell two copies of the same release apart if
they ever differ.

### How files are classified

By **name**, not by the folder they arrived in:
`<L1>_<mode>[_<CEFR level>]_…`, case-insensitive, so both `es_wr_b1_…` of v1 and
`ES_WR_B1_…` of v2.0 work. A CEFR level means a learner text (`learner_es`,
`learner_de`, `learner_cn`, …); no level means a native control (`native_en`,
`native_es`).

Classifying by name rather than by folder is deliberate: the folders mix native
speakers of different languages together, and pooling native English with native
Spanish into one "natives" curve would not be the comparison the figure claims.

### Redistribution

`src/build_reduced_corefl.py` writes numbers only — frequency vectors,
size-matched frequency vectors and vocabulary-growth curves. No COREFL text is
redistributed, the CC BY-NC-ND terms are respected, and `data_raw/corefl/` can be
deleted after the reduction.

---

## 3. PARSEME 1.1 — annotated verbal multi-word expressions

Needed for Table S9 only. Skip it if you do not need that table; nothing else
depends on it.

### Where it comes from

- **Data** — PARSEME shared-task data, release 1.1:
  <https://gitlab.com/parseme/sharedtask-data/-/tree/master/1.1>
  The copy used here is commit `9b47f9465936803bc2a6df857086b6c3d14e909e`.
- **Cite (primary)** — Ramisch, C. *et al.* (2018), *Edition 1.1 of the PARSEME
  Shared Task on Automatic Identification of Verbal Multiword Expressions*,
  Proceedings of LAW-MWE-CxG-2018, 222–240,
  [ACL W18-4925](https://aclanthology.org/W18-4925/).
- **Cite (methodology)** — Savary, A. *et al.* (2017), *The PARSEME Shared Task on
  Automatic Identification of Verbal Multiword Expressions*, Proceedings of the
  13th Workshop on Multiword Expressions, 31–47,
  [doi:10.18653/v1/W17-1704](https://doi.org/10.18653/v1/W17-1704).

### How to download it

Fetch release 1.1 and keep the five language subsets used here — `DE`, `EN`,
`ES`, `FR`, `IT` — placing them as:

```
data_raw/parseme_1_1/DE/{train,dev,test}.cupt
data_raw/parseme_1_1/EN/{train,test}.cupt
data_raw/parseme_1_1/ES/{train,dev,test}.cupt
data_raw/parseme_1_1/FR/{train,dev,test}.cupt
data_raw/parseme_1_1/IT/{train,dev,test}.cupt
```

**English has no `dev.cupt`** — release 1.1 does not ship one, and the check in
notebook 1 accepts whichever splits are present rather than demanding all three.

**Do not use `test.blind.cupt`**: it carries no gold annotation and must not enter
any frequency count. The last cell of notebook 1 checks this layout.

### What the files contain

`.cupt` is CoNLL-U Plus: one token per line, with `PARSEME:MWE` in column 11
carrying the occurrence id and the VMWE category. MWE type frequencies are not a
separate product of the dataset — they are reconstructed by aggregating annotated
occurrences, and the aggregation key has to be stated. Here an **MWE type is the
ordered sequence of its annotated component lemmas, lowercased**, pooled over
train/dev/test.

Gold sizes, summed over the splits:

| language | tokens | VMWE occurrences | underlying treebanks |
| --- | ---: | ---: | --- |
| German | 173,293 | 3,823 | WMT 2015 News Crawl; UD German GSD |
| English | 124,203 | 832 | English Web Treebank; LinES; PUD |
| Spanish | 182,364 | 2,739 | AnCora, AnCora-UD, IXA-UD, UD Spanish |
| French | 528,132 | 5,677 | Sequoia, fr-ud/GSD, ParTUT, PUD |
| Italian | 430,789 | 4,257 | PAISÀ (blogs, Wikinews, Wikipedia) |

Total: 1,438,781 tokens and 17,328 VMWE occurrences.

### The limits this imposes, and why the table is built the way it is

PARSEME 1.1 annotates **verbal** MWEs: verbal idioms, light-verb constructions,
inherently reflexive verbs, verb-particle and multi-verb constructions, and for
some languages inherently adpositional verbs. It is not an exhaustive inventory of
a language's multi-word expressions — nominal, adjectival, adverbial and
functional expressions are not covered in general. Any statement drawn from it is
about *annotated VMWEs*, not about all MWEs.

It is also three orders of magnitude smaller than SPGC. That is why Table S9 does
not compare an MWE exponent with the SPGC exponent directly: it measures four
curves that share one construction and differ only in what is counted or how much
text is available, so that the size effect and the annotation effect can be
separated rather than confounded. See the docstring of `src/mwe_ranks.py`.

### Licence

Redistribution terms differ per language, so the raw files are not committed here:

| language | terms |
| --- | --- |
| EN | VMWE annotations CC BY 4.0; underlying treebanks CC BY-SA |
| DE | VMWE annotations CC BY 4.0; morphosyntactic fields CC BY-NC-SA 3.0 US |
| ES | VMWE annotations CC BY 4.0; mixed sources incl. CC BY, GPLv3, CC BY-NC-SA |
| FR | VMWE annotations CC BY 4.0; UD portion CC BY-NC-SA 4.0; Sequoia LGPL-LR |
| IT | whole dataset CC BY-NC-SA 4.0 |

Each language directory in the release carries its own `README.md` with the
specific treebank citations and full conditions. What this repository commits is
`data_reduced/parseme_mwe.npz`, a frequency spectrum of a few kB.

---

## 4. The annotated concept list

`manifests/annotation_phrases.json` is not a corpus but it is an input, and it
involves a judgement, so it is documented here too.

The file holds 26 concepts, each written the conventional way in each of the five
languages, so that a language which lexicalises a concept as a single compound
(typically German) can be compared with one that spells it out as a phrase.

**The analysis uses the 22 that belong to the period the corpus covers.** SPGC is
dominated by pre-1900 literature, so a concept naming a 20th-century object is
being put to the corpus as a question it cannot answer: its counts are too small
for a rank to mean anything. Three concepts are marked `modern` in the manifest's
`_era` field — `telephone`, `motor car`, `bus stop` — and a fourth, `bus`, goes
with them, on either of two grounds independently. It postdates the corpus as
surely as the other three. And it is a `reference` entry: one of the single words
that exist only to be compared against a compound built from them, so with
`bus stop` gone it has nothing left to be compared against. The other two
reference words, `water` and `bottle`, stay, because `water bottle` stays.

`_era` keeps `bus` under `reference` rather than `modern` because the field
records what a concept is *for* — reference word, period concept, modern concept
— and is read as such by `src/concept_ranks.py` and `src/phrase_ranks.py`. The
exclusion rule is a separate question and lives with the code that applies it.

The exclusion is applied in one place, `MODERN` in `src/concept_ranks.py`, and
`reported_ranks()` uses it for every table and figure alike, which is what keeps
the count at 22 everywhere.

The excluded four stay in the manifest on purpose: the set is what it is by a
stated rule applied to the whole list, not a set trimmed until the numbers
behaved, and that is only checkable if the rule and the discarded entries are
both visible. In 13 of the 22 the languages use different numbers of orthographic
words, up to a difference of two, which is the contrast the analysis exists to
measure.

**The list is a convenience sample, chosen by hand.** It is not drawn from any
defined population and nothing claimed in the paper depends on it being
representative. What makes it usable is that it spans the frequency range and the
compounding contrast, and that nothing is dropped for its outcome: all 22 are
attested in all five languages, on both the weighted and the unweighted curve, so
the period rule is the only filter that removes anything.

A related count, because the two are easy to confuse: `src/phrase_ranks.py`
prints a *coverage* diagnostic of 24 concepts attested in all five languages.
That is the count before the period rule, and it is a different quantity from the
22 the paper reports. Both are correct.

One further boundary bounds what the numbers can mean.

**The five corpora are not translations of one another.** *fauteuil* occurs
8,437 times against *armchair* 449 partly because French Gutenberg novels talk
about drawing rooms more than English ones do. No choice of surface forms can
correct for this, which is precisely why the reported statistic is the **region
of the curve** a concept lands in — its dispersion across languages — and not
equality of ranks.

A related point, stated because it is visible in the data: several English surface
forms in the list are period variants (*looking glass*, *railway station*), which
suppresses rank **correlations** involving English while leaving the dispersion
statistic essentially unchanged. Dispersion is the robust quantity and it is the
one the paper reports; correlations are not.

One tokenisation detail: SPGC joins hyphenated compounds rather than splitting
them (`arm-chair` → `armchair`), verified on bigram counts. Any argument about
orthographic variants has to start from that.
