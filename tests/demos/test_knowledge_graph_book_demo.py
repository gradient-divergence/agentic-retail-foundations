import re
from pathlib import Path

import pytest


def test_ontology_uses_owl_property_types():
    from rdflib import RDF
    from rdflib.namespace import OWL

    from demos.knowledge_graph_book_demo import RetailKnowledgeGraph

    kg = RetailKnowledgeGraph("S1")
    assert (kg.RETAIL.complementsWith, RDF.type, OWL.SymmetricProperty) in kg.graph
    assert (kg.RETAIL.hasSubcategory, RDF.type, OWL.TransitiveProperty) in kg.graph


@pytest.mark.parametrize(
    "filename", ["knowledge_graph_book_demo.py", "sensor_processor_demo.py", "promotion_causal_book_demo.py"]
)
def test_printed_regions_compile(filename):
    path = Path(__file__).resolve().parents[2] / "demos" / filename
    regions = re.findall(
        r"(?ms)^ *# region book:[^\n]+\n(.*?)^ *# endregion(?: book:[^\n]+)?\n", path.read_text()
    )
    assert regions
    compile("".join(regions), filename, "exec", dont_inherit=True)


def test_book_queries_rank_substitutes_complements_and_unpurchased_recommendations():
    from demos.knowledge_graph_book_demo import ProductAttributes, RetailKnowledgeGraph

    kg = RetailKnowledgeGraph("S1")
    for product, name, price, categories in [
        ("A", "Milk", 10.0, ["Dairy"]),
        ("B", "Other Milk", 11.0, ["Dairy"]),
        ("C", "Cream", 9.5, ["Dairy"]),
        ("D", "Expensive Milk", 15.0, ["Dairy"]),
        ("E", "Cookies", 5.0, ["Bakery"]),
        ("F", "Cereal", 5.0, ["Breakfast"]),
    ]:
        kg.add_product(product, name, price, categories, "Acme", ProductAttributes(attributes={}))
    kg.add_product_relationship("A", "substitute", "B", strength=0.9)
    kg.add_product_relationship("A", "complement", "E", strength=0.8)
    kg.add_product_relationship("A", "complement", "C", strength=0.6)
    for index in range(5):
        kg.add_customer_purchase("C1", "A", "2023-01-01T10:00:00", order_id=f"O{index}")
        kg.add_customer_purchase("C1", "F", "2023-01-01T10:00:00", order_id=f"O{index}")
    kg.add_customer_purchase("C1", "B", "2023-01-01T10:00:00")
    substitutes = kg.find_substitutes("A")
    assert [(p.product_id, p.strength) for p in substitutes] == [("B", 0.9), ("C", 0.7)]
    complements = {p.product_id: p.strength for p in kg.find_complementary_products("A")}
    assert complements == {"E": 0.8, "C": 0.6, "F": 0.25}
    assert {p.product_id for p in kg.find_complementary_products("E")} == {"A"}
    recommendations = kg.generate_recommendations("C1")
    assert [(p.product_id, p.relevance_score) for p in recommendations] == [("C", 0.8), ("D", 0.5)]
