# minilm

A 22.7M-parameter GPT-style language model, trained from scratch on TinyStories.

## Generate a story (CPU, no GPU needed)

```
pip install -r requirements.txt
python inference.py --model-dir release --prompt "Once upon a time" --max-new-tokens 256 --temperature 0.8 --top-p 0.9 --seed 42 --device cpu
```

## To run the tests:
 
```
pip install pytest && python -m pytest -q
```