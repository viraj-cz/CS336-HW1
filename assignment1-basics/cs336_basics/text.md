# BPE:
## Unicode1
1. '\x00'
2. String rep: '\x00', printed rep: nothing
3. Nothing, no character is represented

Side Note: this char is a null-char (like in C); in python it is not needed because python is a Unicode aware language. But when interfacing with binary, or any C facing APIs, it is imp to add it for string termination etc.

## Unicode2
1. Most common chars in UTF-8 are represented within 8 bits, so most of our representation will be done in 8 bits, for rare chars it might take a lot more space. On the other hand UTF-16 and UTF-32, take 16 and 32 bits respectively, regardless of how frequent a char is.
2. The reason this is incorrect is because in UTF-8 though most common english chars are represented as 1 byte, that is not the case with all the chars, so a very uncommon char like "🎄" might be decoded incorrectly.
3. "\xf0\x9f" because it is the start of '\xf0\x9f\x98\x82' which is encoding for 😂, logically it makes sense because UTF-8 cannot have conflicting encoding which mean something at 2-byte length and also 4-byte length, because how will it know where the character ends. More concretely:
The first byte's bits:
0xxxxxxx - denotes ASCII
110xxxxx - denotes start of two byte char
1110xxxx - denotes start of three byte char
11110xxx - denotes start of four byte char
10xxxxxx - denotes that this byte is part of the prev seq

# ToDo:
- Apply merge seems to be taking the biggest CPU time (optimize this part)
- Byte object seems heavy in Python, we can save some compute there as well
- I have been stuck on optimizing BPE for a while now, for now I'll move onto the transformer part and come back later
    


# Alpha/Notes
- If we simply use Byte level tokenization, they are "next byte predictors, not even next word/language predictor" in some sense. LLMs ❌, LBMs ✅. There might be strict limitations to what you can end up modelling with language only, so that definitely is a barrier.
- Subword tokenization is a good compromise between byte-tokenization and word-tokenization
- So apparently we pre-tokenize. Normalization & consistency (dog! and dog. should get the same token). Efficiency (BPE is expensive AF for large parts). Handling boundaries( makes sure punctuations etc. get a token instead of them getting merged).
- tiktoken is OpenAI’s fast, specialized tokenizer library for models like GPT-2, GPT-3, GPT-4.

# Questions:
- Gotta look into how the special tokens are handled and if I am handling them correctly


