from bpe import run_train_bpe

if __name__ == "__main__":
    run_train_bpe(
    "/Users/virajchhajed/Desktop/everything/fun/cs336/hw-1/data/TinyStoriesV2-GPT4-valid.txt",
    500,
    ["<|endoftext|>"]
    )