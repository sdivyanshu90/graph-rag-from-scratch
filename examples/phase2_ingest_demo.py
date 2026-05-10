from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graph_rag.chunking import TokenChunker
from graph_rag.extractor import EntityRelationshipExtractor
from graph_rag.graph_store import NetworkXKnowledgeGraph
from graph_rag.ingest import IngestionPipeline
from graph_rag.models import ChunkingConfig


class ScriptedLLMClient:
    def complete_json(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        del user_prompt
        return json.dumps(
            {
                "entities": ["Alice", "Project Atlas", "Bob"],
                "relationships": [
                    {
                        "source": "Alice",
                        "relation": "worked_on",
                        "target": "Project Atlas",
                    },
                    {
                        "source": "Bob",
                        "relation": "managed",
                        "target": "Project Atlas",
                    },
                ],
            }
        )


def main() -> None:
    text = (
        "Alice worked on Project Atlas. Bob managed Project Atlas and reviewed "
        "Alice's delivery timeline."
    )
    pipeline = IngestionPipeline(
        chunker=TokenChunker(ChunkingConfig(chunk_size=64, chunk_overlap=8)),
        extractor=EntityRelationshipExtractor(ScriptedLLMClient()),
        graph_store=NetworkXKnowledgeGraph(),
    )
    report = pipeline.ingest_text(source_id="demo-doc", text=text)

    print("Ingestion report:")
    print(report.model_dump_json(indent=2))

    print("\nGraph stats:")
    print(pipeline.graph_store.stats())

    print("\nNodes with provenance:")
    for node_name, attrs in pipeline.graph_store.graph.nodes(data=True):
        print(f"  - {node_name}: {attrs}")

    print("\nEdges:")
    for source, target, edge_data in pipeline.graph_store.graph.edges(data=True):
        print(f"  - {source} -[{edge_data['relation']}]-> {target} from {edge_data['chunk_id']}")


if __name__ == "__main__":
    main()