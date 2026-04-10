# Gembed Sandbox

A sandbox for benchmarking and exploring the [Gembed](https://github.com/JoelDiaz222/gembed) architecture: an
approach to **in-database embedding generation** that allows database engines to generate embeddings.

## Contents

| Directory                     | Description                                                                                                     |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `benchmark-text-embedders/`   | Benchmark comparing embedding backends (FastEmbed, embed_anything, ORT, gRPC, HTTP)                             |
| `benchmark-pg_gembed-vs-tei/` | Comparison of `pg_gembed` against Text Embeddings Inference (TEI)                                               |
| `benchmark-tei/`              | Standalone TEI benchmark                                                                                        |
| `python-scripts/`             | Benchmark suite of Gembed, written in Python. See the [dedicated README](python-scripts/README.md) for details. |

## License

Licensed under the [Apache License 2.0](./LICENSE).
