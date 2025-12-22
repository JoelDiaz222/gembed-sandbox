```bash
cargo run --release -- \
  --port=50051 \
  --model-id=sentence-transformers/all-MiniLM-L6-v2
```

```bash
cargo run --release --no-default-features --features "http,candle,dynamic-linking" -- \
    --port=8080 \
    --prometheus-port=9001 \
    --model-id=sentence-transformers/all-MiniLM-L6-v2
```
