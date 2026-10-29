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
  - 500 MB run: NOT measured. (TODO: rerun and read python.exe peak in Task
    Manager, or state in the writeup that it was not recorded.)
- Longest token: b' accomplishment' (15 bytes). It makes sense: the GPT-2
  regex attaches the leading space to a word, and "accomplishment" is a
  common word in simple children's stories, so BPE merged it into one token.
- Q1 (cProfile) answer: On the valid file (profiled, 288 s total), merging
  dominated. max() over the pair counts took ~51 s of its own time plus ~53 s
  in its key lambda (99.6M calls), because every one of the ~9,743 merges
  scans all current pairs. Pre-tokenizing (regex .group(), .encode(),
  finditer) took roughly 56 s. Caveat: train_bpe is a single function, so its
  109 s of own time mixes both phases and the split is approximate. On larger
  files pre-tokenizing grows linearly with file size while merging depends
  mostly on the number of unique pre-tokens, so the balance shifts toward
  pre-tokenizing.

## Part 2: Encode and decode
- Tokenizer: tokenizer.py with encode, encode_iterable, decode, from_files.
  Tests in tests/test_tokenizer.py (hand-worked "the cat ate" ->
  [9, 7, 1, 5, 10, 3], round trip, special token, overlapping specials,
  iterable == encode, invalid UTF-8 -> U+FFFD).
  pytest result: 9 passed (3 BPE + 6 tokenizer).
- Dataset encoding (tokenizer trained on first 500 MB of train file):
  - Valid: 5,461,342 tokens from 22,502,601 bytes, 4.12 bytes/token,
    4 s (5.56 MB/s).
  - Train: 540,810,517 tokens from 2,227,753,162 bytes, 4.12 bytes/token,
    720 s (12 min), 3.09 MB/s. Saved as uint16 .npy (~1.08 GB).
  - Speed is higher on the small file; the word cache helps and the
    larger file has more rare words.
- Compression ratio on 10 random stories (valid file, seed 0): 4.14
  bytes/token overall. Per story it ranged from 3.83 to 4.81, so short or
  unusual stories compress a bit worse or better than the average.
- Training will use 5,000 x 32 x 256 = 40.96M tokens, about 7.6% of the
  540.8M-token train array, so the model never sees most of the data.
- Q2 (why uint16): token IDs are at most 9,999 and uint16 holds 0..65,535,
  so each token costs 2 bytes instead of 8 (int64). The train array is about
  1.08 GB instead of ~4.3 GB, and it can be memory-mapped cheaply. It stops
  being enough when the vocab exceeds 65,536 tokens (e.g. large multilingual
  vocabularies of 100k+), where uint32 is needed.

## Part 3
- Slow attention matches F.scaled_dot_product_attention (3-D and 4-D,
  atol 1e-5) and passes a no-future-leak test. pytest: 12 passed.
- Q3: Dot products of random d_k=32 vectors had std 5.57 (theory sqrt(32)=5.66);
  after dividing by sqrt(d_k) the std was 0.98. Without scaling, softmax over
  such scores was nearly one-hot (0.63 and 0.36 on two entries, others
  ~1e-4 to 1e-6), which gives near-zero gradients for most positions. With
  scaling the weights were spread (0.03 to 0.34), keeping gradients healthy.
- RoPE (minilm/model/rope.py): matches a slow full block-diagonal rotation
  matrix (atol 1e-5), preserves vector norm, position 0 is identity, shapes
  preserved with extra batch dims, cos/sin buffers not in state_dict.
  pytest: 18 passed.
- Q4: with the same q and k placed at every position, offset i-j=3 gave
  dot = -3.482756, -3.482755, -3.482755, -3.482756 at (i,j) = (3,0), (10,7),
  (50,47), (100,97). Offsets 0, 1, 5, 20 gave -2.05, -3.04, -1.44, -1.51.
  Reason: rotating q by angle i*w and k by j*w gives q . R((j-i)w) k since
  R(a)^T R(b) = R(b-a); absolute position cancels, only i-j remains.
  Differences in the last digit are float32 rounding.
- CausalSelfAttention (4 bias-free projections, einops head split, RoPE on
  q and k only, SDPA is_causal) and FeedForward (SwiGLU, 3 bias-free
  Linears). Tests: shapes with extra batch dims, no-future-leak, match a
  manual computation using slow_attention, param count of FFN
  (3*512*1344 = 2,064,384). pytest: 25 passed.


## Part 4
- Tests: 30 passed (logits shape (2,17,10000), no future leak in full model,
  initial loss near ln(10000)=9.21, ModuleList/RMSNorm, param count).
- Q5: embedding 5,120,000; per layer attention 1,048,576 + FFN 2,064,384 +
  norms 1,024 = 3,113,984; 4 layers 12,455,936; final norm 512; lm_head
  5,120,000. Total 22,696,448. Embedding + output = 10,240,000 (45%).
- Q6 (256 tokens): per layer projections 536.9M, scores 67.1M, weights*V
  67.1M, FFN 1,057.0M = 1,728.1M. 4 layers 6.91G + lm_head 2.62G = 9.53G.
  FFN is the largest share (44%), then lm_head (27.5%), projections (22.5%),
  attention scores (5.6%).
- Q7 (GPT-2 small, 2-matrix FFN d_ff=3072): per layer projections 4.83G,
  attention 3.22G, FFN 9.66G = 17.72G; 12 layers 212.6G + lm_head 79.05G =
  291.6G. Attention-score share inside a layer rises 7.8% -> 18.2% (scales
  with s^2*d vs s*d^2), FFN share falls 61% -> 54.5%, lm_head share is about
  unchanged (27.5% -> 27.1%).

## Part 5
- pytest: 38 passed, 1 skipped (fp16 test needs a GPU).
- LR warm-up (SGD on mean(w^2), 10 steps): gradient is 0.02*w, so each step
  multiplies w by (1 - 0.02*lr) and the loss by that squared.
  lr=1: loss 26.3 -> 18.3 (factor 0.96/step, slow decay).
  lr=10: 26.3 -> 0.473 (factor 0.64/step, fast decay).
  lr=100: stays 26.3 (factor -1: weights flip sign, no progress).
  lr=1000: 26.3 -> 2.7e24 (factor -19, loss x361/step, diverges).
- Q8 (fp32, 22,696,448 params x 4 bytes): parameters 90.8 MB, gradients
  90.8 MB, AdamW state 181.6 MB (two buffers per parameter: running mean of
  the gradient m and of the squared gradient v), total about 363 MB before
  activations.
- torch.compile speedup (Kaggle T4, fp16, batch 32, ctx 256, data on local
  disk, 200 timed steps after 20 warm-up): no compile 119 and 107 ms/step
  (two runs), compile 86 ms/step, speedup 1.24x vs the faster uncompiled
  run. Run-to-run noise on the uncompiled number is about 10%. Compile adds
  one-off startup time of a minute or two. Inductor warned "Not enough SMs to
  use max_autotune_gemm" (T4 has 40 SMs); harmless, default kernels were used.
- fp16 vs fp32 loss test: passed on Kaggle Tesla T4 (full suite: 39 passed,
  0 skipped). 39 passed in 10.12s on Kaggle.
  On the laptop it is skipped because there is no GPU.

## Part 6
- train.py: all model/optimizer settings from the command line, np.load with
  mmap_mode="r", validation on a fixed set of 20 batches (seeded once, reused
  every eval), CSV logging (step, train_loss, val_loss, lr, grad_norm,
  elapsed), checkpoint every N steps, --resume. Gradients are unscaled
  before clipping (in train_step).
- Check 1 (single batch, tiny model 1.36M params, CPU, lr 3e-3, 300 steps):
  loss 9.34 -> 0.0032. Initial loss matches ln(10000)=9.21. Grad norm
  peaked at 1.01 (step 20) and decayed to 0.002, no NaN. LR warmed up to
  3e-3 by step 20 and decayed to 3e-4 at step 300. Val loss rose from 7.35
  to 10.39 as the model memorised the one batch, which shows validation is
  held out from the training batch. Plot: figures/single.png
- Check 2 (resume, tiny model, 200 steps, stopped at 100): after resume the
  lr continued on the same cosine curve (5.16e-4 at step 110, 1.00e-4 at
  step 200), with no loss spike. Val loss at step 150/200: 5.3520/5.2694
  resumed vs 5.3515/5.2695 uninterrupted. Training losses differ slightly
  per step because batches are sampled differently after resume (same trend,
  small per-step noise). Exact reproduction is covered by
  test_checkpoint_resume_reproduces_losses. Plot: figures/resume.png
- Check 3 (grad norm, first 100 steps): 0.49 -> 0.93 peak at step 40 ->
  0.47, no NaN, not growing.
- LR pilots (300 steps, fp16, warmup 5%, cosine to lr/10), final val loss:
  1e-4: 3.9985, 3e-4: 3.3399, 1e-3: 2.8493, 3e-3: 2.8284, 1e-2: 3.4414.
  3e-2 and 1e-1 blew up (non-finite loss at step <30 and <20 respectively;
  at 1e-1 the loss was 17.35 and grad norm 18.3 at step 10). Clipping to 1.0
  did not prevent it, because Adam's update size scales with lr, not with
  the gradient magnitude.
- Q9: The best rate (3e-3) is a factor of 10 below the first rate that broke
  (3e-2), and the next rate above it (1e-2) was stable but clearly worse.
  So the best rate is below the edge, but not "just below": 1e-2 sits in
  between and is stable yet inefficient. Statement holds loosely: the
  optimum is within about an order of magnitude below the blow-up point.
  3e-3 and 1e-3 were within noise (0.02) of each other.
- How rates were chosen: log-spaced sweep (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
  across the handout's range; since none broke, extended upward (3e-2, 1e-1)
  until training diverged. Figure: figures/pilots.png

## Part 7
- Main run: lr 3e-3, 5,000 steps, batch 32, ctx 256 = 40,960,000 tokens,
  fp16 + torch.compile, Kaggle T4. Warmup 250 steps, cosine to 3e-4,
  reached exactly at step 5000.
- Final val loss 1.7435 (fixed 20-batch validation set), perplexity
  e^1.7435 = 5.72. Beats the 2.0 stretch target. Train loss about 1.7,
  grad norm about 0.3 at the end, no NaNs or spikes.
- Saw 41M of 541M available tokens, so no data was repeated.
- Training time: 472.1 s (7.9 min) on a Kaggle T4, from the last `elapsed`
  value in runs/main/log.csv (includes torch.compile startup and the 20
  validation passes).
- Loss curves: figures/main_loss.png (loss vs steps and vs elapsed time).
- generate(): temperature + top-p, stops at <|endoftext|>, feeds only the
  last 256 tokens. Greedy decoding is deterministic (test passes).
- Stories: prompt "Once upon a time", min_new_tokens=256 (EOS suppressed
  until 256 new tokens; unforced, this model ends stories at about 125-180
  tokens). Settings: (temp 0.7, top_p 0.9) seeds 1 and 2, (temp 1.0,
  top_p 0.95) seed 1. Full text in stories.txt.
- Q10: Stories are fluent at the sentence level: correct grammar, consistent
  character names, proper dialogue punctuation, simple TinyStories plots.
  Two things that change quality:
  (1) Temperature/top-p. At 0.7 text is coherent but repetitive ("play with
  it" repeated, "be careful" loop in seed 1). At 1.0 more low-probability
  tokens get through and errors appear: seed 2 has "the 'fet triangle!'",
  "had chose to pay attention to the attic"; seed 1 ends in broken
  fragments and repeated "The moral of the story is..." lines.
  (2) Length vs. training. Without forcing, stories end at 125-180 tokens.
  Past that, forced continuation drifts into repeated morals and logic slips
  ("put the rock on the rock", "the rock could talk"). That is a limit of a
  22.7M-parameter model trained on 41M tokens, not a code bug.


## Part 8: Packaging
- release/ contains exactly the four files the handout lists: model.pt
  (weights only, about 90 MB), config.json, vocab.json, merges.txt.
- README command:
  python inference.py --model-dir release --prompt "Once upon a time"
  --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
- Same --seed gives an identical story (two runs with seed 42 matched
  word for word).
- Acceptance test (fresh git clone into a new folder, new venv, pip install
  -r requirements.txt, README command): passed on CPU. It printed the same
  seed-42 story as in the main project (identical text, same torch version).
- Q11 measured speed: 43.3 and 50.4 tokens/s (two runs, 184 tokens in 4.3 s
  and 3.7 s), CPU only (--device cpu, PyTorch fp32), on an Intel Core 5 210H
  (8 cores), 16 GB RAM, Windows. The two runs differ by about 15%, which I
  treat as CPU noise. The story ended at 184 tokens because the model
  emitted <|endoftext|> before the 256-token limit.
- Q11 repeated work (estimate, not measured): with a 4-token prompt and 184
  generated tokens, no-cache decoding processes about 4+5+...+188, roughly
  17,700 token positions, versus 188 with a cache, so about 99% of the
  per-token work is repeated. A KV cache would cut the projections,
  feed-forward and output layer to one token per step. Attention scores
  still scale with the context length per step, so that part only goes from
  quadratic to linear in total. Expected saving is several times in total
  time, but I did not implement or time a cache. Caveat: once the text
  passes the 256-token context, the window slides and a simple cache would
  need extra handling.


## Bugs found vs model limits

### Bugs / mistakes in my setup or code (found and fixed)
- ModuleNotFoundError: No module named 'minilm' when running pytest. Cause:
  project root was not on the import path. Fix: added conftest.py in the root.
- Saved merges.txt with latin-1 text split on a space. Ambiguous because a
  token can contain a space byte. Fix: switched vocab.json and merges.txt to
  hex encoding.
- .gitignore was empty at first, so git status listed data/ (2 GB train
  file). Fixed before the first commit.
- Profiling overhead: the 288 s timing was inflated by cProfile/tracemalloc.
  Real training time is 36 s.
- My own time estimates were off (BPE on 500 MB: predicted 15-30 min, took
  5.2 min; train encode: predicted ~7 min, took 12 min). Reported measured
  values only.
- runs/ was committed to git by mistake. Fix: git rm -r --cached runs and
  added runs/ to .gitignore.
- Data loading bottleneck on Kaggle: get_batch took 606 ms/step from the
  memory-mapped file on /kaggle/input (model step: 111 ms). Fix: copied
  train.npy and valid.npy to /kaggle/temp (local disk); get_batch then took
  1 ms. The first torch.compile timing (2101 vs 953 ms, 2.21x) was dominated
  by disk reads and is not reported.

### Limits (not bugs)
- Tokenizer trained on the first 500 MB of the 2 GB train file only.
- Peak memory for the 500 MB BPE run was not recorded.
- Pilots are only 300 steps, so 3e-3 vs 1e-3 is within noise and may differ
  over 5,000 steps.
- fp16 non-finite loss at lr 3e-2 and 1e-1: cause not separated between
  fp16 overflow and genuine divergence (no fp32 rerun was done).