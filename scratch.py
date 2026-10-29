# Part 1 warm-up experiments (scratch, not graded code)

print("--- Q1: chr(0) ---")
print(repr(chr(0)))
print(chr(0))
print("this is a test" + chr(0) + "string")
print(repr("this is a test" + chr(0) + "string"))

print("\n--- Q2: encodings ---")
for s in ["hello", "नमस्ते", "😀"]:
    for enc in ["utf-8", "utf-16", "utf-32"]:
        b = s.encode(enc)
        print(f"{s!r:10} {enc:7} len={len(b):3} bytes={list(b)}")

print("\n--- Q3: byte-by-byte decoding ---")
b = "नमस्ते".encode("utf-8")
print("bytes:", list(b))
try:
    print("".join(bytes([x]).decode("utf-8") for x in b))
except UnicodeDecodeError as e:
    print("ERROR:", e)
print("all at once:", b.decode("utf-8"))

print("\n--- Q4: invalid two-byte sequences ---")
for pair in [bytes([0xC0, 0x80]), bytes([0xFF, 0xFF]), bytes([0xC3, 0x28])]:
    try:
        print(list(pair), "->", repr(pair.decode("utf-8")))
    except UnicodeDecodeError as e:
        print(list(pair), "-> ERROR:", e)