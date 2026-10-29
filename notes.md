# Notes

## Part 1 warm-ups

### Q1: chr(0)
Observed: repr(chr(0)) -> '\x00'; print shows nothing visible, so
"this is a test" + chr(0) + "string" prints as "this is a teststring".
Answer: chr(0) is the null character. repr() shows it as an explicit
escape ('\x00'), so you can see it sitting inside the string, while print()
sends the raw invisible character to the terminal and it looks like nothing
is there.

### Q2: UTF-8 vs UTF-16 vs UTF-32
Observed: "hello" -> 5 bytes (utf-8), 12 (utf-16), 24 (utf-32).
"नमस्ते" -> 18 / 14 / 28 bytes. "😀" -> 4 / 6 / 8 bytes.
Answer: UTF-8 is not always the smallest (Hindi is smaller in UTF-16), but
for English it is by far the most compact. UTF-16/32 add a byte-order mark
and pad ASCII characters with zero bytes, which a tokenizer would waste
merges learning to remove. UTF-8 also gives a fixed base alphabet of exactly
256 byte values, so any text in any language can be represented and nothing
is ever "unknown".

### Q3: Byte-by-byte decoding
Observed: crashes at byte 0xe0, "unexpected end of data".
Answer: Many characters take more than one byte in UTF-8, and a single byte
from the middle (or start) of a multi-byte character is not a valid character
on its own. Decoding must see all bytes of a character together. It works for
plain ASCII, which is why the bug is easy to miss. Example that breaks:
"नमस्ते".encode("utf-8") (each character is 3 bytes), or any emoji.

### Q4: Invalid two-byte sequence
Example: bytes([0xC3, 0x28]) -> "invalid continuation byte".
Answer: 0xC3 = 11000011 is a valid start of a two-byte character, so the next
byte must look like 10xxxxxx. But 0x28 = 00101000 does not start with 10, so
the sequence is invalid. (bytes([0xFF, 0xFF]) also fails: 0xFF = 11111111 is
never a valid start byte.)

## Part 1: BPE
- Hand-worked merges (toy corpus low x5, lower x2, widest x3, newest x6):
  (s,t) (e,st) (o,w) (l,ow) (w,est) (n,e). The first merge is a tie
  between (s,t) and (e,s) at 9 each; the tie-break (larger tuple) picks
  (s,t). My test_toy_example reproduces this order and passes.
- Training time:
  - Valid file (~20 MB): 36 s without profiling. The first run took 288 s
    only because cProfile and tracemalloc add heavy overhead.
  - Train file: first 500 MB only (cut at a story boundary, as the
    assignment allows): 314 s (5.2 min), vocab size 10,000.
- Peak memory:
  - Valid file: 90 MB (Python allocations only, via tracemalloc).
  - 500 MB run: NOT measured. (TODO: rerun and read the python.exe peak in
    Task Manager, or state that it was not recorded.)
- Longest token: b' accomplishment' (15 bytes). It makes sense: the GPT-2
  regex attaches the leading space to a word, and "accomplishment" is a
  common word in simple children's stories, so BPE merged it into one token.
- Q1 (cProfile) answer: On the valid file (profiled, 288 s total), merging
  dominated. max() over the pair counts took ~51 s of its own time plus ~53 s
  in its key lambda (99.6M calls), because every one of the ~9,743 merges
  scans all current pairs. Pre-tokenizing (regex match .group(), .encode(),
  finditer) took roughly 56 s. Caveat: train_bpe is a single function, so its
  109 s of own time mixes both phases and the split is approximate. Splitting
  it into two helper functions would separate them cleanly. On larger files
  pre-tokenizing grows linearly with file size while merging depends mostly
  on the number of unique pre-tokens, so the balance shifts toward
  pre-tokenizing.

## Part 2
- (fill in after encoding: compression ratio, bytes/s, encode time, Q2 uint16)

## Bugs found vs model limits

### Bugs / mistakes in my setup or code (found and fixed)
- ModuleNotFoundError: No module named 'minilm' when running pytest. Cause:
  project root was not on the import path. Fix: added conftest.py in the root.
- Saved merges.txt with latin-1 text split on a space. This is ambiguous
  because a token can contain a space byte. Fix: switched vocab.json and
  merges.txt to hex encoding.
- .gitignore was empty at first, so git status listed data/ (2 GB train
  file). Fixed before the first commit.
- Profiling overhead: the 288 s timing was inflated by cProfile/tracemalloc.
  Real training time is 36 s.

### Limits (not bugs)
- Tokenizer trained on the first 500 MB of the 2 GB train file only.
- Peak memory numbers are Python-only for the valid file.