# External Popularity Preprocessing

Use `build_external_popularity.py` for public reproduction. It provides five
parameterized subcommands:

```text
prepare          Extract entities and required question-answer pairs
wikidata         Query Wikidata IDs and sitelink counts (resumable)
wikipedia-index  Scan Wikipedia parquet files and build entity → document IDs
aggregate        Compute single-entity occurrence and pairwise co-occurrence
annotate         Attach all external fields to response rows
```

The complete commands, Wikipedia snapshot, schemas, and methodological notes
are documented in
[`../../docs/BUILD_POPULARITY_FEATURES.md`](../../docs/BUILD_POPULARITY_FEATURES.md).

The other Python files in this directory are retained as historical provenance
for the original experiment workspace. They contain hard-coded cluster-era
paths and should not be used as the public entry point.

