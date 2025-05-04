"""
RetailKnowledgeGraph: A semantic knowledge graph for retail intelligence.

This module provides the RetailKnowledgeGraph class, which enables structured representation, querying, and reasoning over retail entities, relationships, and events. It supports both local RDF graphs and optional external SPARQL endpoints.

Key Capabilities:
- Product, customer, store, and promotion modeling
- Relationship and event tracking (e.g., purchases, substitutes, complements)
- Semantic queries for recommendations, substitutes, and complements
- Export/import in standard RDF formats
- Extensible for integration with other agentic retail systems

Adapted from the in-notebook implementation in sensor-networks-and-cognitive-systems.py.
"""

from typing import Any, List, Dict
from datetime import datetime
from rdflib import Graph, Literal, BNode, Namespace, RDF, URIRef
from rdflib.namespace import RDFS, XSD


class RetailKnowledgeGraph:
    """
    A structured, semantic knowledge graph for retail environments.

    This class enables the construction, querying, and reasoning over a retail knowledge graph, supporting both local RDF graphs and optional external SPARQL endpoints.

    Core Entities:
    - Products, Customers, Stores, Categories, Promotions
    - Relationships: substitutes, complements, accessories, variants, purchases
    - Events: customer purchases, product recommendations

    Example usage:
        kg = RetailKnowledgeGraph(store_id="S001")
        kg.add_product("P123", "Milk", 2.99, ["Dairy"], brand="Acme")
        kg.add_customer_purchase("C001", "P123", "2023-06-01T10:00:00", quantity=2)
        substitutes = kg.find_substitutes("P123")
    """

    def __init__(self, store_id: str, graph_uri: str | None = None):
        """Initialize the retail knowledge graph."""
        self.store_id = store_id
        self.graph = Graph()
        # Namespaces for retail domain
        self.RETAIL = Namespace("http://retail.example.org/ontology#")
        self.PRODUCT = Namespace("http://retail.example.org/product/")
        self.CATEGORY = Namespace("http://retail.example.org/category/")
        self.STORE = Namespace("http://retail.example.org/store/")
        self.CUSTOMER = Namespace("http://retail.example.org/customer/")
        # Bind namespaces for easier querying
        self.graph.bind("retail", self.RETAIL)
        self.graph.bind("product", self.PRODUCT)
        self.graph.bind("category", self.CATEGORY)
        self.graph.bind("store", self.STORE)
        self.graph.bind("customer", self.CUSTOMER)
        self._load_ontology()
        self.sparql_endpoint = None # Initialize as None
        
        # Only attempt SPARQL setup if graph_uri is provided
        if graph_uri:
            SPARQLWrapper = None # Local variable for check
            JSON = None
            try:
                # Attempt import only if needed
                from SPARQLWrapper import SPARQLWrapper as SPARQLWrapper_lib, JSON as JSON_lib
                SPARQLWrapper = SPARQLWrapper_lib
                JSON = JSON_lib
            except ImportError:
                print("SPARQLWrapper not installed, SPARQL endpoint disabled.")
                # sparql_endpoint remains None
            
            # If import succeeded, try to create instance
            if SPARQLWrapper and JSON:
                try:
                    self.sparql_endpoint = SPARQLWrapper(graph_uri)
                    self.sparql_endpoint.setReturnFormat(JSON)
                except Exception as e:
                    print(f"Failed to initialize SPARQLWrapper for {graph_uri}: {e}")
                    self.sparql_endpoint = None # Ensure it's None on failure

    def _load_ontology(self):
        """Load the retail domain ontology into the graph."""
        # Core classes
        self.graph.add((self.RETAIL.Product, RDF.type, RDFS.Class))
        self.graph.add((self.RETAIL.Category, RDF.type, RDFS.Class))
        self.graph.add((self.RETAIL.Store, RDF.type, RDFS.Class))
        self.graph.add((self.RETAIL.Customer, RDF.type, RDFS.Class))
        self.graph.add((self.RETAIL.Location, RDF.type, RDFS.Class))
        # Properties
        self.graph.add((self.RETAIL.name, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.price, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.hasCategory, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.locatedIn, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.hasBrand, RDF.type, RDF.Property))
        # Relationship properties
        self.graph.add((self.RETAIL.isSubstituteFor, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.complementsWith, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.isAccessoryFor, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.isVariantOf, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.purchased, RDF.type, RDF.Property))
        # Property definitions
        self.graph.add((self.RETAIL.isSubstituteFor, RDFS.domain, self.RETAIL.Product))
        self.graph.add((self.RETAIL.isSubstituteFor, RDFS.range, self.RETAIL.Product))
        self.graph.add((self.RETAIL.complementsWith, RDFS.domain, self.RETAIL.Product))
        self.graph.add((self.RETAIL.complementsWith, RDFS.range, self.RETAIL.Product))
        # Symmetric and transitive properties
        self.graph.add(
            (self.RETAIL.complementsWith, RDF.type, self.RETAIL.SymmetricProperty)
        )
        self.graph.add(
            (self.RETAIL.hasSubcategory, RDF.type, self.RETAIL.TransitiveProperty)
        )

    def add_product(
        self,
        product_id: str,
        name: str,
        price: float,
        category_ids: list[str],
        brand: str | None = None,
        attributes: dict[str, str] | None = None,
    ) -> URIRef:
        """Add a product to the knowledge graph."""
        if not product_id or not name:
            raise ValueError("product_id and name are required.")
        product_uri = self.PRODUCT[product_id]
        self.graph.add((product_uri, RDF.type, self.RETAIL.Product))
        self.graph.add((product_uri, self.RETAIL.name, Literal(name)))
        self.graph.add(
            (product_uri, self.RETAIL.price, Literal(price, datatype=XSD.decimal))
        )
        if brand:
            self.graph.add((product_uri, self.RETAIL.hasBrand, Literal(brand)))
        if category_ids:
            for category_id in category_ids:
                if not category_id:
                    continue
                category_uri = self.CATEGORY[category_id]
                self.graph.add((category_uri, RDF.type, self.RETAIL.Category))
                self.graph.add((product_uri, self.RETAIL.hasCategory, category_uri))
        if attributes:
            for attr_name, attr_value in attributes.items():
                if not attr_name or attr_value is None:
                    continue
                attr_property = self.RETAIL[attr_name.replace(" ", "_")]
                if (attr_property, RDF.type, RDF.Property) not in self.graph:
                    self.graph.add((attr_property, RDF.type, RDF.Property))
                self.graph.add((product_uri, attr_property, Literal(str(attr_value))))
        return product_uri

    def add_product_relationship(
        self,
        source_product_id: str,
        relationship_type: str,
        target_product_id: str,
        strength: float | None = None,
        metadata: dict[str, str] | None = None,
    ):
        """Add a relationship between products."""
        source_uri = self.PRODUCT[source_product_id]
        target_uri = self.PRODUCT[target_product_id]
        if relationship_type == "substitute":
            relation = self.RETAIL.isSubstituteFor
        elif relationship_type == "complement":
            relation = self.RETAIL.complementsWith
        elif relationship_type == "accessory":
            relation = self.RETAIL.isAccessoryFor
        elif relationship_type == "variant":
            relation = self.RETAIL.isVariantOf
        else:
            relation = self.RETAIL[relationship_type]
        self.graph.add((source_uri, relation, target_uri))
        if strength is not None:
            if 0.0 <= strength <= 1.0:
                relation_node = BNode()
                self.graph.add((relation_node, RDF.type, RDF.Statement))
                self.graph.add((relation_node, RDF.subject, source_uri))
                self.graph.add((relation_node, RDF.predicate, relation))
                self.graph.add((relation_node, RDF.object, target_uri))
                self.graph.add(
                    (
                        relation_node,
                        self.RETAIL.strength,
                        Literal(strength, datatype=XSD.decimal),
                    )
                )
        if metadata:
            for key, value in metadata.items():
                meta_property = self.RETAIL[key]
                self.graph.add((relation_node, meta_property, Literal(value)))

    def add_customer_purchase(
        self,
        customer_id: str,
        product_id: str,
        timestamp: str,
        quantity: int = 1,
        order_id: str | None = None,
        channel: str | None = "in_store",
    ):
        """Record a customer purchase in the knowledge graph."""
        customer_uri = self.CUSTOMER[customer_id]
        product_uri = self.PRODUCT[product_id]
        purchase_node = BNode()
        self.graph.add((purchase_node, RDF.type, self.RETAIL.Purchase))
        self.graph.add((purchase_node, self.RETAIL.hasCustomer, customer_uri))
        self.graph.add((purchase_node, self.RETAIL.hasProduct, product_uri))
        try:
            datetime.fromisoformat(timestamp)
            self.graph.add(
                (
                    purchase_node,
                    self.RETAIL.timestamp,
                    Literal(timestamp, datatype=XSD.dateTime),
                )
            )
        except ValueError:
            pass
        if quantity > 0:
            self.graph.add(
                (
                    purchase_node,
                    self.RETAIL.quantity,
                    Literal(quantity, datatype=XSD.integer),
                )
            )
        if order_id:
            self.graph.add((purchase_node, self.RETAIL.orderID, Literal(order_id)))
        self.graph.add((purchase_node, self.RETAIL.channel, Literal(channel)))
        self.graph.add((customer_uri, self.RETAIL.purchased, product_uri))

    def find_substitutes(
        self, product_id: str, max_results: int = 5
    ) -> list[dict[str, Any]]:
        """Find substitute products for a given product."""
        query = f"""
        PREFIX retail: <http://retail.example.org/ontology#>
        PREFIX product: <http://retail.example.org/product/>
        SELECT ?substitute ?name ?price ?brand ?strength
        WHERE {{
            {{ product:{product_id} retail:isSubstituteFor ?substitute .
                OPTIONAL {{ 
                    ?stmt rdf:type rdf:Statement ;
                          rdf:subject product:{product_id} ;
                          rdf:predicate retail:isSubstituteFor ;
                          rdf:object ?substitute ;
                          retail:strength ?strength .
                }}
            }}
            UNION
            {{ ?substitute retail:isSubstituteFor product:{product_id} .
                OPTIONAL {{ 
                    ?stmt rdf:type rdf:Statement ;
                          rdf:subject ?substitute ;
                          rdf:predicate retail:isSubstituteFor ;
                          rdf:object product:{product_id} ;
                          retail:strength ?strength .
                }}
            }}
            UNION
            {{ product:{product_id} retail:hasCategory ?category .
                ?substitute retail:hasCategory ?category .
                product:{product_id} retail:price ?originalPrice .
                ?substitute retail:price ?price .
                FILTER (?substitute != product:{product_id})
                FILTER (?price >= ?originalPrice * 0.8 && ?price <= ?originalPrice * 1.2)
                BIND(0.7 as ?strength)
            }}
            ?substitute retail:name ?name .
            ?substitute retail:price ?price .
            ?substitute retail:hasBrand ?brand .
            BIND(COALESCE(?strength, 1.0) as ?strength)
        }}
        ORDER BY DESC(?strength) ?price
        LIMIT {max_results}
        """
        results = self._execute_query(query)
        substitutes = []
        for row in results:
            try:
                substitute_uri = str(row["substitute"])
                substitute_id = substitute_uri.split("/")[-1]
                substitutes.append(
                    {
                        "product_id": substitute_id,
                        "name": str(row["name"]),
                        "price": float(row["price"]),
                        "brand": str(row["brand"]),
                        "strength": float(row["strength"]),
                    }
                )
            except (KeyError, ValueError, TypeError):
                continue
        return substitutes

    def find_complementary_products(
        self, product_id: str, max_results: int = 5
    ) -> list[dict[str, Any]]:
        """Find products that complement a given product."""
        query = f"""
        PREFIX retail: <http://retail.example.org/ontology#>
        PREFIX product: <http://retail.example.org/product/>
        SELECT ?complement ?name ?price ?brand ?strength ?relation_type
        WHERE {{
            {{ product:{product_id} retail:complementsWith ?complement .
                BIND("complement" AS ?relation_type)
                OPTIONAL {{ 
                    ?stmt rdf:type rdf:Statement ;
                          rdf:subject product:{product_id} ;
                          rdf:predicate retail:complementsWith ;
                          rdf:object ?complement ;
                          retail:strength ?strength .
                }}
            }}
            UNION
            {{ ?complement retail:isAccessoryFor product:{product_id} .
                BIND("accessory" AS ?relation_type)
                OPTIONAL {{ 
                    ?stmt rdf:type rdf:Statement ;
                          rdf:subject ?complement ;
                          rdf:predicate retail:isAccessoryFor ;
                          rdf:object product:{product_id} ;
                          retail:strength ?strength .
                }}
            }}
            UNION
            {{ SELECT ?complement (COUNT(*) as ?count) ("co_purchase" AS ?relation_type)
                WHERE {{
                    ?purchase1 retail:hasProduct product:{product_id} ;
                              retail:hasCustomer ?customer ;
                              retail:orderID ?order .
                    ?purchase2 retail:hasProduct ?complement ;
                              retail:hasCustomer ?customer ;
                              retail:orderID ?order .
                    FILTER(?complement != product:{product_id})
                }}
                GROUP BY ?complement
                HAVING(COUNT(*) >= 5)
            }}
            ?complement retail:name ?name .
            ?complement retail:price ?price .
            ?complement retail:hasBrand ?brand .
            BIND(
                IF(?relation_type = "co_purchase", 
                   ?count / 20, 
                   COALESCE(?strength, 1.0))
                AS ?strength
            )
        }}
        ORDER BY DESC(?strength) ?relation_type
        LIMIT {max_results}
        """
        results = self._execute_query(query)
        complements = []
        for row in results:
            try:
                complement_uri = str(row["complement"])
                complement_id = complement_uri.split("/")[-1]
                complements.append(
                    {
                        "product_id": complement_id,
                        "name": str(row["name"]),
                        "price": float(row["price"]),
                        "brand": str(row["brand"]),
                        "strength": float(row["strength"]),
                        "relation_type": str(row["relation_type"]),
                    }
                )
            except (KeyError, ValueError, TypeError):
                continue
        return complements

    def _execute_query(self, query_str: str) -> list[dict]:
        """Execute a SPARQL query against the knowledge graph."""
        if self.sparql_endpoint:
            try:
                self.sparql_endpoint.setQuery(query_str)
                sparql_results_raw = self.sparql_endpoint.query().convert()
                if isinstance(sparql_results_raw, dict):
                    bindings = sparql_results_raw.get("results", {}).get("bindings", []) # type: ignore[union-attr]
                    return bindings if isinstance(bindings, list) else []
                else:
                    print(f"Unexpected SPARQL result type: {type(sparql_results_raw)}")
                    return [] 
            except Exception as e:
                print(f"SPARQL query failed: {e}")
                return []
        else:
            local_results: List[Dict[str, Any]] = [] # type: ignore[no-redef] # Ignore potential redef if mypy confused
            try:
                qres = self.graph.query(query_str)
                binding_vars = [str(v) for v in getattr(qres, 'vars', [])]

                for row in qres:
                    result_dict: Dict[str, Any] = {}
                    try:
                        result_dict = row.asdict() # type: ignore[union-attr]
                    except AttributeError:
                        if isinstance(row, tuple) and len(row) == len(binding_vars):
                            for i, var_name in enumerate(binding_vars):
                                value: Any = row[i]
                                result_dict[var_name] = value
                        
                    if result_dict: 
                        local_results.append(result_dict) # type: ignore[union-attr]
                
            except Exception as e:
                print(f"Local RDF query failed: {e}")
            return local_results # type: ignore[return-value]

    def generate_recommendations(
        self,
        customer_id: str,
        current_context: dict[str, Any] | None = None,
        max_results: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate personalized recommendations for a customer."""
        query = f"""
        PREFIX retail: <http://retail.example.org/ontology#>
        PREFIX customer: <http://retail.example.org/customer/>
        SELECT DISTINCT ?product ?name ?price ?brand ?score
        WHERE {{
            {{ customer:{customer_id} retail:purchased ?purchasedProduct .
                ?purchasedProduct retail:hasCategory ?category .
                ?product retail:hasCategory ?category .
                FILTER(?product != ?purchasedProduct)
                BIND(0.5 AS ?baseScore)
                ?product retail:name ?name .
                ?product retail:price ?price .
                ?product retail:hasBrand ?brand .
            }}
            OPTIONAL {{
                customer:{customer_id} retail:purchased ?otherProduct .
                ?product retail:complementsWith ?otherProduct .
                BIND(0.3 AS ?complementBoost)
            }}
            BIND(COALESCE(?baseScore, 0) + COALESCE(?complementBoost, 0) AS ?score)
        }}
        ORDER BY DESC(?score) ?name
        LIMIT {max_results}
        """
        results = self._execute_query(query)
        recommendations = []
        for row in results:
            try:
                product_uri = str(row["product"])
                product_id = product_uri.split("/")[-1]
                recommendations.append(
                    {
                        "product_id": product_id,
                        "name": str(row["name"]),
                        "price": float(row["price"]),
                        "brand": str(row["brand"]),
                        "relevance_score": float(row["score"]),
                    }
                )
            except (KeyError, ValueError, TypeError):
                continue
        return recommendations

    def export_graph(self, file_path: str, format: str = "turtle"):
        """Export the knowledge graph in the specified format."""
        try:
            self.graph.serialize(destination=file_path, format=format)
        except Exception:
            pass

    def load_graph(self, source: str | bytes, format: str = "turtle"):
        """Load data into the knowledge graph."""
        try:
            self.graph.parse(data=source, format=format)
        except Exception:
            pass

    def clear_graph(self, preserve_ontology: bool = True):
        """Clear all data from the graph except the ontology."""
        if preserve_ontology:
            ontology_triples = [
                triple
                for triple in self.graph
                # Convert subject to string before calling startswith
                if str(triple[0]).startswith(str(self.RETAIL)) # Cast both to str 
                and triple[1] in (RDF.type, RDFS.domain, RDFS.range)
            ]
            self.graph = Graph()
            self.graph.bind("retail", self.RETAIL)
            self.graph.bind("product", self.PRODUCT)
            self.graph.bind("category", self.CATEGORY)
            self.graph.bind("store", self.STORE)
            self.graph.bind("customer", self.CUSTOMER)
            for triple in ontology_triples:
                self.graph.add(triple)
        else:
            self.graph = Graph()
            self.graph.bind("retail", self.RETAIL)
            self.graph.bind("product", self.PRODUCT)
            self.graph.bind("category", self.CATEGORY)
            self.graph.bind("store", self.STORE)
            self.graph.bind("customer", self.CUSTOMER)
