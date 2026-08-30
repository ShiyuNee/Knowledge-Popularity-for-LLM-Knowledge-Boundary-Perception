from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "prepare_before_analysis" / "build_external_popularity.py"
SPEC = importlib.util.spec_from_file_location("build_external_popularity", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ExternalPopularityPipelineTest(unittest.TestCase):
    def test_prepare_index_and_aggregate(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            responses = root / "responses.jsonl"
            responses.write_text(
                json.dumps(
                    {
                        "dataset": "movies",
                        "question": "Who is the director of the movie Example Film",
                        "reference": ["Alice Smith"],
                        "response": "Bob Jones",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            entities = root / "entities.jsonl"
            pairs = root / "pairs.jsonl"
            MODULE.command_prepare(
                argparse.Namespace(
                    inputs=[str(responses)],
                    entities_output=str(entities),
                    pairs_output=str(pairs),
                )
            )

            wikipedia = root / "wikipedia.parquet"
            pq.write_table(
                pa.table(
                    {
                        "id": ["1", "2", "3"],
                        "text": [
                            "Example Film was directed by Alice Smith. Alice Smith returned.",
                            "A page about Example Film and Bob Jones.",
                            "Alice Smith has a separate biography.",
                        ],
                    }
                ),
                wikipedia,
            )
            index = root / "entity_documents.parquet"
            MODULE.command_wikipedia_index(
                argparse.Namespace(
                    entities=str(entities),
                    wikipedia=[str(wikipedia)],
                    id_column="id",
                    text_column="text",
                    batch_size=2,
                    output=str(index),
                )
            )

            single = root / "single.json"
            cooccurrence = root / "cooccurrence.json"
            MODULE.command_aggregate(
                argparse.Namespace(
                    index=str(index),
                    pairs=str(pairs),
                    single_output=str(single),
                    cooccurrence_output=str(cooccurrence),
                )
            )
            self.assertEqual(
                json.loads(single.read_text()),
                {"alice smith": 2, "bob jones": 1, "example film": 2},
            )
            self.assertEqual(
                json.loads(cooccurrence.read_text()),
                {"example film": {"alice smith": 1, "bob jones": 1}},
            )

            wikidata = root / "wikidata.jsonl"
            MODULE.write_jsonl(
                wikidata,
                [
                    {"entity": "example film", "found": True, "sitelinks": 9},
                    {"entity": "alice smith", "found": True, "sitelinks": 5},
                    {"entity": "bob jones", "found": False, "sitelinks": None},
                ],
            )
            annotated = root / "annotated.jsonl"
            MODULE.command_annotate(
                argparse.Namespace(
                    input=str(responses),
                    wikidata=str(wikidata),
                    single=str(single),
                    cooccurrence=str(cooccurrence),
                    output=str(annotated),
                )
            )
            row = list(MODULE.read_jsonl(annotated))[0]
            self.assertEqual(
                {key: row[key] for key in ("qpop", "gt_pop", "gene_pop", "q_occ", "gt_occ", "gene_occ", "gt_coo", "gene_coo")},
                {"qpop": 9, "gt_pop": 5, "gene_pop": 0, "q_occ": 2, "gt_occ": 2, "gene_occ": 1, "gt_coo": 1, "gene_coo": 1},
            )
            self.assertFalse(row["gene_pop_found"])

    def test_wikidata_output_is_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entities = root / "entities.jsonl"
            MODULE.write_jsonl(
                entities,
                [{"entity": "alice smith", "surface_forms": ["Alice Smith"], "roles": ["ground_truth"]}],
            )
            output = root / "wikidata.jsonl"
            args = argparse.Namespace(
                entities=str(entities),
                output=str(output),
                endpoint="https://example.invalid/sparql",
                user_agent="test",
                timeout=1,
                retries=0,
                retry_delay=0,
                delay=0,
                resume=True,
            )
            with patch.object(MODULE, "query_wikidata", return_value=("Q123", 17)) as query:
                MODULE.command_wikidata(args)
                MODULE.command_wikidata(args)
            self.assertEqual(query.call_count, 1)
            self.assertEqual(
                list(MODULE.read_jsonl(output)),
                [{
                    "entity": "alice smith",
                    "found": True,
                    "matched_surface": "Alice Smith",
                    "sitelinks": 17,
                    "wikidata_id": "Q123",
                }],
            )


if __name__ == "__main__":
    unittest.main()
