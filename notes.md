# Notes

## Part 1 warm-ups

### Q1: chr(0)
Observed: repr(chr(0)) -> '\x00'; print shows nothing visible, so
"this is a test" + chr(0) + "string" prints as "this is a teststring".
Answer: (write 1-2 sentences: what does repr show that print hides?)

### Q2: UTF-8 vs UTF-16 vs UTF-32
Observed: "hello" -> 5 bytes (utf-8), 12 (utf-16), 24 (utf-32).
"नमस्ते" -> 18 / 14 / 28 bytes. "😀" -> 4 / 6 / 8 bytes.
Answer: (why train on UTF-8 anyway? think: alphabet size, English, zero bytes)

### Q3: Byte-by-byte decoding
Observed: crashes at byte 0xe0, "unexpected end of data".
Answer: (why is it wrong? give an input where it breaks)

### Q4: Invalid two-byte sequence
Example: bytes([0xC3, 0x28]) -> "invalid continuation byte".
Answer: (explain using the bit patterns of a start byte vs continuation byte)

## Part 1: BPE
- Hand-worked merges (toy corpus): 
- Training time:
- Peak memory:
- Longest token:
- Q1 (cProfile) answer:

## Bugs found vs model limits
(log these as you go, the writeup needs them)