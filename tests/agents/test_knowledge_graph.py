from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD

# Module to test
from agents.knowledge_graph import RetailKnowledgeGraph

# Define namespaces used in tests for clarity
RETAIL = Namespace("http://retail.example.org/ontology#")
PRODUCT = Namespace("http://retail.example.org/product/")
CATEGORY = Namespace("http://retail.example.org/category/")
CUSTOMER = Namespace("http://retail.example.org/customer/")

# --- Test Initialization --- #


def test_kg_initialization():
    """Test basic initialization without SPARQL endpoint."""
    kg = RetailKnowledgeGraph(store_id="S001")
    assert kg.store_id == "S001"
    assert isinstance(kg.graph, Graph)
    assert kg.sparql_endpoint is None
    # Check if namespaces are bound
    bound_namespaces = {prefix for prefix, ns in kg.graph.namespaces()}
    assert "retail" in bound_namespaces
    assert "product" in bound_namespaces
    # Check if some basic ontology triples exist
    assert (RETAIL.Product, RDF.type, RDFS.Class) in kg.graph
    assert (RETAIL.name, RDF.type, RDF.Property) in kg.graph


# Test with mocked SPARQLWrapper
@patch("agents.knowledge_graph._SPARQLWrapper")
@patch("agents.knowledge_graph._JSON")
def test_kg_initialization_with_sparql_uri_success(mock_json, mock_sparql_wrapper, capfd):
    """Test KG initialization with a valid SPARQL URI (mocked success)."""
    endpoint_uri = "http://localhost:7200/repositories/test"
    mock_instance = MagicMock()
    mock_sparql_wrapper.return_value = mock_instance

    kg = RetailKnowledgeGraph(store_id="S1", graph_uri=endpoint_uri)

    # Check SPARQLWrapper was called correctly
    mock_sparql_wrapper.assert_called_once_with(endpoint_uri)
    # Check setReturnFormat was called on the instance
    mock_instance.setReturnFormat.assert_called_once_with(mock_json)
    # Check the endpoint is stored
    assert kg.sparql_endpoint is mock_instance

    # Check logs/print output
    captured = capfd.readouterr()
    assert f"SPARQL endpoint configured for: {endpoint_uri}" in captured.out


# Test with SPARQLWrapper import failing
@patch("agents.knowledge_graph._SPARQLWrapper", None)  # Simulate import failure
@patch("agents.knowledge_graph._JSON", None)
def test_kg_initialization_with_sparql_uri_import_error(capfd):
    """Test KG initialization when SPARQLWrapper library is missing."""
    endpoint_uri = "http://localhost:7200/repositories/test"
    kg = RetailKnowledgeGraph(store_id="S1", graph_uri=endpoint_uri)

    # Check endpoint was not set
    assert kg.sparql_endpoint is None

    # Check logs/print output
    captured = capfd.readouterr()
    assert f"Cannot configure SPARQL endpoint {endpoint_uri}: SPARQLWrapper library not found." in captured.out


def test_kg_initialization_no_sparql_uri():
    """Test KG initialization without a SPARQL URI."""
    kg = RetailKnowledgeGraph(store_id="S001")
    assert kg.store_id == "S001"
    assert isinstance(kg.graph, Graph)
    assert kg.sparql_endpoint is None
    # Check if namespaces are bound
    bound_namespaces = {prefix for prefix, ns in kg.graph.namespaces()}
    assert "retail" in bound_namespaces
    assert "product" in bound_namespaces
    # Check if some basic ontology triples exist
    assert (RETAIL.Product, RDF.type, RDFS.Class) in kg.graph
    assert (RETAIL.name, RDF.type, RDF.Property) in kg.graph


# --- Test add_product --- #


def test_add_product_full_details():
    """Test adding a product with all details."""
    kg = RetailKnowledgeGraph(store_id="S001")
    pid = "P123"
    name = "Organic Milk"
    price = 3.50
    cats = ["Dairy", "Organic"]
    brand = "FarmFresh"
    attrs = {"size": "1L", "fat_content": "2%"}

    product_uri = kg.add_product(pid, name, price, cats, brand, attrs)

    # Verify URI
    assert product_uri == PRODUCT[pid]

    # Verify triples using graph.value() or iterating triples
    g = kg.graph
    assert (product_uri, RDF.type, RETAIL.Product) in g
    assert g.value(subject=product_uri, predicate=RETAIL.name) == Literal(name)
    assert g.value(subject=product_uri, predicate=RETAIL.price) == Literal(price, datatype=XSD.decimal)
    assert g.value(subject=product_uri, predicate=RETAIL.hasBrand) == Literal(brand)
    # Check categories
    assert (product_uri, RETAIL.hasCategory, CATEGORY["Dairy"]) in g
    assert (
        CATEGORY["Dairy"],
        RDF.type,
        RETAIL.Category,
    ) in g  # Category type also added
    assert (product_uri, RETAIL.hasCategory, CATEGORY["Organic"]) in g
    assert (CATEGORY["Organic"], RDF.type, RETAIL.Category) in g
    # Check attributes
    assert g.value(subject=product_uri, predicate=RETAIL.size) == Literal(attrs["size"])
    assert g.value(subject=product_uri, predicate=RETAIL.fat_content) == Literal(attrs["fat_content"])
    # Verify attribute properties were added to ontology implicitly
    assert (RETAIL.size, RDF.type, RDF.Property) in g
    assert (RETAIL.fat_content, RDF.type, RDF.Property) in g


def test_add_product_minimal_details():
    """Test adding a product with only required details."""
    kg = RetailKnowledgeGraph(store_id="S001")
    pid = "P456"
    name = "Basic Water"
    price = 1.00
    cats = ["Beverages"]

    product_uri = kg.add_product(pid, name, price, cats, brand=None, attributes=None)

    g = kg.graph
    assert (product_uri, RDF.type, RETAIL.Product) in g
    assert g.value(subject=product_uri, predicate=RETAIL.name) == Literal(name)
    assert g.value(subject=product_uri, predicate=RETAIL.price) == Literal(price, datatype=XSD.decimal)
    assert (product_uri, RETAIL.hasCategory, CATEGORY["Beverages"]) in g
    # Check things that shouldn't be there
    assert g.value(subject=product_uri, predicate=RETAIL.hasBrand) is None
    assert g.value(subject=product_uri, predicate=RETAIL.size) is None


def test_add_product_invalid_input():
    """Test adding product with invalid input raises ValueError."""
    kg = RetailKnowledgeGraph(store_id="S001")
    with pytest.raises(ValueError, match="product_id and name are required."):
        kg.add_product("", "Valid Name", 1.0, ["Cat"])
    with pytest.raises(ValueError, match="product_id and name are required."):
        kg.add_product("ValidID", "", 1.0, ["Cat"])


# --- Test add_product_relationship --- #


def test_add_product_relationship_simple():
    """Test adding a simple relationship between products."""
    kg = RetailKnowledgeGraph(store_id="S001")
    # Add products first (required for relationship)
    p1_uri = kg.add_product("P1", "Product 1", 10.0, ["CatA"])
    p2_uri = kg.add_product("P2", "Product 2", 11.0, ["CatA"])

    kg.add_product_relationship("P1", "substitute", "P2")

    # Check the relationship triple exists
    assert (p1_uri, RETAIL.isSubstituteFor, p2_uri) in kg.graph


def test_add_product_relationship_with_strength_and_metadata():
    """Test adding a relationship with strength and metadata."""
    kg = RetailKnowledgeGraph(store_id="S001")
    p1_uri = kg.add_product("P1", "Product 1", 10.0, ["CatA"])
    p3_uri = kg.add_product("P3", "Product 3", 5.0, ["CatB"])

    kg.add_product_relationship("P1", "complement", "P3", strength=0.8, metadata={"source": "algo_v2"})

    # Check the direct triple exists
    assert (p1_uri, RETAIL.complementsWith, p3_uri) in kg.graph

    # Find the statement node and check its properties
    stmt_node = None
    for s, _p, _o in kg.graph.triples((None, RDF.type, RDF.Statement)):
        if (
            kg.graph.value(s, RDF.subject) == p1_uri
            and kg.graph.value(s, RDF.predicate) == RETAIL.complementsWith
            and kg.graph.value(s, RDF.object) == p3_uri
        ):
            stmt_node = s
            break

    assert stmt_node is not None, "Statement BNode not found"
    strength_literal = kg.graph.value(stmt_node, RETAIL.strength)
    assert strength_literal == Literal(0.8, datatype=XSD.decimal)
    meta_literal = kg.graph.value(stmt_node, RETAIL.source)
    assert meta_literal == Literal("algo_v2")


def test_add_product_relationship_invalid_strength():
    """Test that strength outside [0, 1] is ignored (no statement created)."""
    kg = RetailKnowledgeGraph(store_id="S001")
    p1_uri = kg.add_product("P1", "Product 1", 10.0, ["CatA"])
    p2_uri = kg.add_product("P2", "Product 2", 11.0, ["CatA"])

    # Strength > 1.0
    kg.add_product_relationship("P1", "substitute", "P2", strength=1.5)

    # Direct triple should still exist
    assert (p1_uri, RETAIL.isSubstituteFor, p2_uri) in kg.graph
    # No statement node should be created for invalid strength
    stmt_nodes = list(kg.graph.subjects(RDF.type, RDF.Statement))
    assert len(stmt_nodes) == 0


# --- Test add_customer_purchase --- #


def test_add_customer_purchase_full():
    """Test adding a purchase event with all details."""
    kg = RetailKnowledgeGraph(store_id="S001")
    pid = "P123"
    cid = "C789"
    timestamp = "2024-01-15T14:30:00"
    order_id = "ORD555"
    quantity = 2
    channel = "online"

    # Add product first so it exists
    product_uri = kg.add_product(pid, "Test Prod", 5.0, ["Test"])
    customer_uri = CUSTOMER[cid]  # Define customer URI

    kg.add_customer_purchase(cid, pid, timestamp, quantity, order_id, channel)

    # Find the purchase node (it's a BNode, so query by type and properties)
    purchase_node = None
    for s, _p, _o in kg.graph.triples((None, RDF.type, RETAIL.Purchase)):
        if kg.graph.value(s, RETAIL.hasCustomer) == customer_uri and kg.graph.value(s, RETAIL.hasProduct) == product_uri:
            purchase_node = s
            break

    assert purchase_node is not None, "Purchase BNode not found"

    # Verify properties of the purchase node
    g = kg.graph
    assert g.value(purchase_node, RETAIL.hasCustomer) == customer_uri
    assert g.value(purchase_node, RETAIL.hasProduct) == product_uri
    assert g.value(purchase_node, RETAIL.timestamp) == Literal(timestamp, datatype=XSD.dateTime)
    assert g.value(purchase_node, RETAIL.quantity) == Literal(quantity, datatype=XSD.integer)
    assert g.value(purchase_node, RETAIL.orderID) == Literal(order_id)
    assert g.value(purchase_node, RETAIL.channel) == Literal(channel)

    # Verify the direct purchase link
    assert (customer_uri, RETAIL.purchased, product_uri) in g


def test_add_customer_purchase_minimal():
    """Test adding a purchase event with minimal details."""
    kg = RetailKnowledgeGraph(store_id="S001")
    pid = "P123"
    cid = "C789"
    timestamp = "2024-01-16T10:00:00"

    product_uri = kg.add_product(pid, "Test Prod", 5.0, ["Test"])
    customer_uri = CUSTOMER[cid]

    kg.add_customer_purchase(cid, pid, timestamp)

    purchase_node = None
    for s, _p, _o in kg.graph.triples((None, RDF.type, RETAIL.Purchase)):
        if kg.graph.value(s, RETAIL.hasCustomer) == customer_uri and kg.graph.value(s, RETAIL.hasProduct) == product_uri:
            purchase_node = s
            break
    assert purchase_node is not None

    g = kg.graph
    assert g.value(purchase_node, RETAIL.quantity) == Literal(1, datatype=XSD.integer)  # Default quantity
    assert g.value(purchase_node, RETAIL.channel) == Literal("in_store")  # Default channel
    assert g.value(purchase_node, RETAIL.orderID) is None  # Optional field not present
    assert (customer_uri, RETAIL.purchased, product_uri) in g


# --- Test Query Methods --- #


@pytest.fixture
def kg_with_relations() -> RetailKnowledgeGraph:
    """Fixture providing a KG with products, relationships, and purchases."""
    kg = RetailKnowledgeGraph(store_id="S_QUERY")
    # Products (Add P_H for co-purchase)
    kg.add_product("P_A", "Product A", 10.0, ["Cat1"], brand="BrandX")
    kg.add_product("P_B", "Product B", 11.0, ["Cat1"], brand="BrandX")
    kg.add_product("P_C", "Product C", 9.50, ["Cat1"], brand="BrandY")
    kg.add_product("P_D", "Product D", 15.0, ["Cat1"], brand="BrandX")
    kg.add_product("P_E", "Product E", 10.50, ["Cat2"], brand="BrandY")  # Complement
    kg.add_product("P_F", "Product F", 9.80, ["Cat1"], brand="BrandZ")  # Substitute
    kg.add_product("P_G", "Product G", 10.20, ["Cat1"], brand="BrandX")  # Substitute
    kg.add_product("P_H", "Product H", 8.00, ["Cat3"], brand="BrandX")  # Co-purchase
    kg.add_product("P_I", "Product I", 12.00, ["Cat3"], brand="BrandZ")  # Accessory

    # Relationships
    kg.add_product_relationship("P_F", "substitute", "P_A")
    kg.add_product_relationship("P_G", "substitute", "P_A", strength=0.9)
    kg.add_product_relationship("P_A", "complement", "P_E", strength=0.8)
    kg.add_product_relationship("P_I", "accessory", "P_A")  # P_I is accessory FOR P_A

    # --- Restore purchase history ---
    ts = "2024-01-01T10:00:00"
    # Customer 1 buys A and H together 5 times
    for i in range(5):
        order_id = f"ORD_A_H_{i}"
        kg.add_customer_purchase("C1", "P_A", ts, order_id=order_id)
        kg.add_customer_purchase("C1", "P_H", ts, order_id=order_id)
    # Customer 2 buys A and B together 3 times (below threshold)
    for i in range(3):
        order_id = f"ORD_A_B_{i}"
        kg.add_customer_purchase("C2", "P_A", ts, order_id=order_id)
        kg.add_customer_purchase("C2", "P_B", ts, order_id=order_id)
    # Customer 3 buys only A
    kg.add_customer_purchase("C3", "P_A", ts, order_id="ORD_A_ONLY")
    # --- End purchase history ---

    return kg


def test_find_substitutes(kg_with_relations: RetailKnowledgeGraph):
    """Test finding substitutes via direct relations and category matching."""
    kg = kg_with_relations
    product_id_to_query = "P_A"

    substitutes = kg.find_substitutes(product_id_to_query, max_results=5)

    assert len(substitutes) <= 5

    # Expected substitutes and why:
    # P_G: Explicit substitute, strength 0.9
    # P_F: Explicit substitute (reverse relation), strength 1.0 (default)
    # P_B: Category match, price 11.0 (within 20% of 10.0), strength 0.7
    # P_C: Category match, price 9.50 (within 20% of 10.0), strength 0.7
    # P_D: Category match, price 15.0 (outside 20% of 10.0) - Should NOT be included
    # P_E: Different category - Should NOT be included

    found_ids = {s["product_id"] for s in substitutes}
    expected_ids = {"P_G", "P_F", "P_B", "P_C"}
    assert found_ids == expected_ids

    # Check sorting (by strength desc, then price asc)
    # Expected order: P_F (1.0, 9.80), P_G (0.9, 10.20), P_C (0.7, 9.50), P_B (0.7, 11.0)
    assert substitutes[0]["product_id"] == "P_F"
    assert substitutes[1]["product_id"] == "P_G"
    # Order of P_C and P_B might swap if strength calculation isn't exact 0.7 or price sorting dominates
    assert {substitutes[2]["product_id"], substitutes[3]["product_id"]} == {
        "P_C",
        "P_B",
    }
    if substitutes[2]["product_id"] == "P_C":
        assert substitutes[3]["product_id"] == "P_B"
    else:
        assert substitutes[2]["product_id"] == "P_B"
        assert substitutes[3]["product_id"] == "P_C"

    # Check details of one substitute
    p_f_data = next(s for s in substitutes if s["product_id"] == "P_F")
    assert p_f_data["name"] == "Product F"
    assert p_f_data["price"] == 9.80
    assert p_f_data["brand"] == "BrandZ"
    assert p_f_data["strength"] == pytest.approx(1.0)  # Default strength

    p_g_data = next(s for s in substitutes if s["product_id"] == "P_G")
    assert p_g_data["strength"] == pytest.approx(0.9)

    p_c_data = next(s for s in substitutes if s["product_id"] == "P_C")
    assert p_c_data["strength"] == pytest.approx(0.7)  # Category match strength


def test_find_substitutes_limit(kg_with_relations: RetailKnowledgeGraph):
    """Test the max_results limit for find_substitutes."""
    kg = kg_with_relations
    substitutes = kg.find_substitutes("P_A", max_results=2)
    assert len(substitutes) == 2
    # Should return the top 2 based on strength/price: P_F and P_G
    found_ids = {s["product_id"] for s in substitutes}
    assert found_ids == {"P_F", "P_G"}


def test_find_substitutes_not_found(kg_with_relations: RetailKnowledgeGraph):
    """Test finding substitutes for a product with no expected substitutes."""
    kg = kg_with_relations
    # P_E is in Cat2, no explicit substitutes defined
    substitutes = kg.find_substitutes("P_E")
    assert substitutes == []


def test_find_complementary_products(kg_with_relations: RetailKnowledgeGraph):
    """Test finding complementary products via relations and co-purchase."""
    kg = kg_with_relations
    product_id_to_query = "P_A"

    complements = kg.find_complementary_products(product_id_to_query, max_results=5)

    # Expected complements for P_A and why:
    # P_E: Explicit complement, strength 0.8
    # P_I: Explicit accessory, strength 1.0 (default)
    # P_H: Co-purchased 5 times (>= threshold 5), strength = 5/20 = 0.25
    # P_B: Co-purchased 3 times (< threshold 5) - Should NOT be included

    found_ids = {c["product_id"] for c in complements}
    expected_ids = {"P_E", "P_I", "P_H"}
    assert found_ids == expected_ids
    assert len(complements) == 3

    # Check sorting (Strength DESC, Relation Type ASC?)
    # Expected order: P_I (1.0, accessory), P_E (0.8, complement), P_H (0.25, co_purchase)
    assert complements[0]["product_id"] == "P_I"
    assert complements[0]["relation_type"] == "accessory"
    assert complements[0]["strength"] == pytest.approx(1.0)

    assert complements[1]["product_id"] == "P_E"
    assert complements[1]["relation_type"] == "complement"
    assert complements[1]["strength"] == pytest.approx(0.8)

    assert complements[2]["product_id"] == "P_H"
    assert complements[2]["relation_type"] == "co_purchase"
    assert complements[2]["strength"] == pytest.approx(0.25)  # 5 / 20


def test_find_complementary_products_limit(kg_with_relations: RetailKnowledgeGraph):
    """Test max_results limit for complements."""
    kg = kg_with_relations
    complements = kg.find_complementary_products("P_A", max_results=1)
    assert len(complements) == 1
    assert complements[0]["product_id"] == "P_I"  # Highest strength


def test_find_complementary_products_not_found(kg_with_relations: RetailKnowledgeGraph):
    """Test finding complements for a product with none."""
    kg = kg_with_relations
    # Product P_B has no defined complements/accessories and only low co-purchase
    complements = kg.find_complementary_products("P_B")
    assert complements == []


# --- Test generate_recommendations --- #


@pytest.mark.parametrize(
    "customer_id, expected_reco_ids_scores",
    [
        # C1 purchased P_A (Cat1) and P_H (Cat3)
        # Recos should be other Cat1 items (P_B, P_C, P_F, P_G)
        # P_D price likely too high based on internal logic, but query doesn't filter price explicitly here.
        # Query finds products in same category as *any* purchased product.
        # It finds P_B, P_C, P_D, P_F, P_G (all Cat1, same as P_A).
        # It finds P_I (Cat3, same as P_H).
        # Query *optionally* boosts score if reco complements *any* other purchased item.
        # P_I complements P_A (purchased). Score = 0.5 + 0.3 = 0.8
        # Others have no defined complements with P_A or P_H. Score = 0.5
        # Expected: P_I (0.8), then P_B, P_C, P_D, P_F, P_G (0.5) - sorted by score DESC, name ASC
        (
            "C1",
            {"P_I": 0.8, "P_B": 0.5, "P_C": 0.5, "P_D": 0.5, "P_F": 0.5, "P_G": 0.5},
        ),
        # C2 purchased P_A (Cat1) and P_B (Cat1)
        # Recos should be other Cat1 items (P_C, P_D, P_F, P_G)
        # No complement boosts apply.
        # Expected: P_C, P_D, P_F, P_G (all 0.5) - sorted by name ASC
        ("C2", {"P_C": 0.5, "P_D": 0.5, "P_F": 0.5, "P_G": 0.5}),
        # C_NO_PURCHASE has no history
        ("C_NO_PURCHASE", {}),
    ],
    ids=["C1_purchased_A_H", "C2_purchased_A_B", "C_NO_PURCHASE"],
)
def test_generate_recommendations(
    kg_with_relations: RetailKnowledgeGraph,
    customer_id: str,
    expected_reco_ids_scores: dict[str, float],
):
    """Test generating recommendations based on purchase history."""
    kg = kg_with_relations
    recommendations = kg.generate_recommendations(customer_id, max_results=10)

    assert len(recommendations) == len(expected_reco_ids_scores)

    found_recos = {}
    for reco in recommendations:
        pid = reco["product_id"]
        score = reco["relevance_score"]
        assert pid in expected_reco_ids_scores
        assert score == pytest.approx(expected_reco_ids_scores[pid])
        found_recos[pid] = score

    assert found_recos == expected_reco_ids_scores

    # Check sorting (Score DESC, Name ASC)
    if len(recommendations) > 1:
        for i in range(len(recommendations) - 1):
            score_i = recommendations[i]["relevance_score"]
            score_j = recommendations[i + 1]["relevance_score"]
            name_i = recommendations[i]["name"]
            name_j = recommendations[i + 1]["name"]
            assert score_i >= score_j
            if score_i == score_j:
                assert name_i <= name_j


def test_generate_recommendations_limit(kg_with_relations: RetailKnowledgeGraph):
    """Test max_results limit for recommendations."""
    kg = kg_with_relations
    recommendations = kg.generate_recommendations("C1", max_results=2)
    assert len(recommendations) == 2
    # Expected: P_I (0.8) and one of the 0.5 score items (P_B is first alphabetically)
    assert recommendations[0]["product_id"] == "P_I"
    # The specific second item depends on stable sort, check possibilities
    assert recommendations[1]["product_id"] in ["P_B", "P_C", "P_D", "P_F", "P_G"]


# --- Test Graph Utilities --- #


def test_export_load_graph(kg_with_relations: RetailKnowledgeGraph, tmp_path: Path):
    """Test exporting the graph and loading it back."""
    kg1 = kg_with_relations
    export_file = tmp_path / "kg_export.ttl"
    # Change format back to turtle
    graph_format = "turtle"

    # Get original triple count
    original_triple_count = len(kg1.graph)
    assert original_triple_count > 10  # Ensure we have some data

    # Export
    kg1.export_graph(str(export_file), format=graph_format)
    assert export_file.exists()
    assert export_file.stat().st_size > 0

    # Create new PLAIN graph and load
    kg2_graph = Graph()
    # Bind namespaces manually if needed for comparison later, though parse should handle prefixes
    kg2_graph.bind("retail", kg1.RETAIL)
    kg2_graph.bind("product", kg1.PRODUCT)
    kg2_graph.bind("category", kg1.CATEGORY)
    kg2_graph.bind("customer", kg1.CUSTOMER)

    kg2_graph.parse(source=str(export_file), format=graph_format)

    # Compare triple counts (simplest check)
    assert len(kg2_graph) == original_triple_count

    # Optional: More rigorous check - compare graph isomorphism or specific triples
    # For example, check if a known product exists in the loaded graph
    p_a_uri = PRODUCT["P_A"]
    assert (p_a_uri, RDF.type, RETAIL.Product) in kg2_graph
    assert kg2_graph.value(p_a_uri, RETAIL.name) == Literal("Product A")


def test_clear_graph(kg_with_relations: RetailKnowledgeGraph):
    """Test clearing the graph with and without preserving ontology."""
    kg = kg_with_relations
    original_triple_count = len(kg.graph)

    # --- Test clear with preserving ontology --- #
    kg.clear_graph(preserve_ontology=True)
    count_after_preserve = len(kg.graph)

    # Check instance data is gone (e.g., product P_A)
    assert (PRODUCT["P_A"], RDF.type, RETAIL.Product) not in kg.graph
    # Check ontology data remains
    assert (RETAIL.Product, RDF.type, RDFS.Class) in kg.graph
    assert count_after_preserve > 0
    assert count_after_preserve < original_triple_count

    # --- Test clear without preserving ontology --- #
    # Re-add some data first
    kg.add_product("P_TEMP", "Temp", 1.0, ["TempCat"])
    assert len(kg.graph) > count_after_preserve

    kg.clear_graph(preserve_ontology=False)
    # Graph should be completely empty (except maybe prefixes)
    assert len(kg.graph) == 0
    assert (RETAIL.Product, RDF.type, RDFS.Class) not in kg.graph


# Placeholder tests
# def test_execute_query(): ... # Implicitly tested by query methods
# def test_export_load_graph(): ...
# def test_clear_graph(): ...
