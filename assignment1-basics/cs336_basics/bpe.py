import regex as re
import os
from collections import defaultdict
from multiprocessing import Pool
from itertools import chain
from pretokenization_example import find_chunk_boundaries
from tqdm import tqdm

def init_worker():
    global PAT
    PAT = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


def worker(args):
        input_path, start, end, special_tokens = args
        global PAT
        with open(input_path, "rb") as f:
            f.seek(start)
            pre_chunk = f.read(end - start).decode("utf-8", errors="ignore")

            if special_tokens:
                pattern = "|".join(re.escape(tok) for tok in special_tokens)
                pre_chunk = [c for c in re.split(pattern, pre_chunk) if c]
            else:
                pre_chunk = [pre_chunk]
            
        # 
        parts = []
        for c in pre_chunk:
            for m in PAT.finditer(c):
                tok = m.group(0)
                parts.append(tuple(bytes([b]) for b in tok.encode("utf8")))
        return parts


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    vocab = {i: bytes([i]) for i in range(256)}
    curr_vocab_size = 256
    for st in special_tokens:
        vocab[curr_vocab_size] = st.encode("utf-8")
        curr_vocab_size += 1
    
    merges = []
    args_list = []

    with open(input_path, "rb") as f:
        num_processes = os.cpu_count() or 1
        print(f"num processes: {num_processes}")
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            args_list.append((input_path, start, end, special_tokens))

    num_chunks = max(0, len(args_list))
    num_processes = min(num_processes, max(1, num_chunks))

    with Pool(processes=num_processes, initializer=init_worker) as pool:
        results = pool.map(worker, args_list)
    
    parts = list(chain.from_iterable(results))
                
    loop_counter = 0
    total = vocab_size - curr_vocab_size
    pbar = tqdm(total=total, desc="BPE")

    merge_dict = defaultdict(int)
    max_bucket = defaultdict(set)
    current_max = 0

    for part in parts:
        total_letters = len(part)
        k = 0
        while (k+1) < total_letters:
            pair = (part[k], part[k+1])
            old = merge_dict[pair]
            if old > 0:
                max_bucket[old].discard(pair)
            new = old + 1
            merge_dict[pair] = new
            max_bucket[new].add(pair)
            if new > current_max:
                current_max = new
            k += 1

    while (curr_vocab_size < vocab_size):
        
        if not merge_dict:
            break
        
        while current_max > 0 and not max_bucket[current_max]:
            current_max -= 1
        if current_max == 0:
            break
        best_pair = max(max_bucket[current_max])
        merges.append((best_pair[0], best_pair[1]))
        new_token = best_pair[0] + best_pair[1]
        vocab[curr_vocab_size] = new_token
        curr_vocab_size += 1

        def apply_merge(part):
            j = 0
            out = []
            while j+1 < len(part):
                if (part[j], part[j+1]) == best_pair:
                    out.append(new_token)
                    j += 2
                else:
                    out.append(part[j])
                    j += 1
                
            if j < len(part):
                out.append(part[j])
            return tuple(out)

        new_parts = []
        changed_parts_new = []
        changed_parts_old = []
        for p in parts:
            new_p = apply_merge(p)
            new_parts.append(new_p)
            if new_p != p:
                changed_parts_old.append(p)
                changed_parts_new.append(new_p)

        if new_parts == parts:
            break
        parts = new_parts

        for part in changed_parts_old:
            total_letters = len(part)
            k = 0
            while (k+1) < total_letters:
                pair = (part[k], part[k+1])
                old = merge_dict[pair]
                if old > 0:
                    max_bucket[old].discard(pair)
                    new = old - 1
                    if new > 0:
                        merge_dict[pair] = new
                        max_bucket[new].add(pair)
                    else:
                        merge_dict.pop(pair, None)
                k += 1
            
        for part in changed_parts_new:
            total_letters = len(part)
            k = 0
            while (k+1) < total_letters:
                pair = (part[k], part[k+1])
                old = merge_dict[pair]
                if old > 0:
                    max_bucket[old].discard(pair)
                new = old + 1
                merge_dict[pair] = new
                max_bucket[new].add(pair)
                if new > current_max:
                    current_max = new
                k += 1
        
        pbar.update(1)
        loop_counter+= 1
        
    pbar.close()
    return (vocab, merges)
