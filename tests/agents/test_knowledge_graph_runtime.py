import re
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["agents/knowledge_graph.py", "demos/knowledge_graph_book_demo.py"])
def graph_module(request, monkeypatch):
    # Remote-result decoding needs no RDF engine. Real local queries are tested separately.
    monkeypatch.setitem(
        sys.modules,
        "rdflib",
        SimpleNamespace(
            RDF=object(), BNode=object, Graph=object, Literal=object, Namespace=object, URIRef=object
        ),
    )
    monkeypatch.setitem(
        sys.modules, "rdflib.namespace", SimpleNamespace(OWL=object(), RDFS=object(), XSD=object())
    )
    monkeypatch.setitem(sys.modules, "SPARQLWrapper", SimpleNamespace(JSON=object(), SPARQLWrapper=object))
    return runpy.run_path(str(ROOT / request.param))


def test_remote_bindings_are_decoded_for_recommendations(graph_module):
    class Endpoint:
        def setQuery(self, query):
            assert "SELECT" in query

        def query(self):
            return self

        def convert(self):
            return {
                "results": {
                    "bindings": [
                        {
                            "product": {"type": "uri", "value": "http://retail.example.org/product/P2"},
                            "name": {"type": "literal", "value": "Milk"},
                            "price": {"type": "literal", "value": "2.99"},
                            "brand": {"type": "literal", "value": "Acme"},
                            "score": {"type": "literal", "value": "0.8"},
                        }
                    ]
                }
            }

    analyzer_type = graph_module["RetailKnowledgeGraph"]
    kg = analyzer_type.__new__(analyzer_type)
    kg.sparql_endpoint = Endpoint()
    results = kg.generate_recommendations("C1")
    assert len(results) == 1
    result = results[0] if isinstance(results[0], dict) else results[0].model_dump()
    assert result == {
        "product_id": "P2",
        "name": "Milk",
        "price": 2.99,
        "brand": "Acme",
        "relevance_score": 0.8,
    }


def test_book_local_graph_import_does_not_require_sparqlwrapper(graph_module, monkeypatch):
    monkeypatch.setitem(sys.modules, "SPARQLWrapper", None)
    runpy.run_path(str(ROOT / "demos/knowledge_graph_book_demo.py"))


@pytest.mark.parametrize("method", ["find_substitutes", "find_complementary_products"])
def test_relationship_queries_declare_rdf_prefix_for_remote_endpoints(graph_module, method):
    class Endpoint:
        def setQuery(self, query):
            for expression, target in re.findall(r"BIND\s*\((.*?)\s+AS\s+\?(\w+)\s*\)", query, re.I | re.S):
                assert not re.search(rf"\?{re.escape(target)}\b", expression), (
                    "BIND reuses its input variable"
                )
            assert "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>" in query

        def query(self):
            return self

        def convert(self):
            return {
                "results": {
                    "bindings": [
                        {
                            key: {"type": "literal", "value": value}
                            for key, value in {
                                "substitute": "http://retail.example.org/product/P2",
                                "complement": "http://retail.example.org/product/P2",
                                "name": "Milk",
                                "price": "2.99",
                                "brand": "Acme",
                                "strength": "0.8",
                                "final_strength": "0.8",
                                "relation_type": "complement",
                            }.items()
                        }
                    ]
                }
            }

    analyzer_type = graph_module["RetailKnowledgeGraph"]
    kg = analyzer_type.__new__(analyzer_type)
    kg.sparql_endpoint = Endpoint()
    results = getattr(kg, method)("P1")
    assert len(results) == 1
    result = results[0] if isinstance(results[0], dict) else results[0].model_dump()
    assert result["product_id"] == "P2"
    assert result["strength"] == 0.8
