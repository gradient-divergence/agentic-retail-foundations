# region book:knowledge-graph-imports

from pydantic import BaseModel
from rdflib import RDF, BNode, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDFS, XSD
from SPARQLWrapper import JSON, SPARQLWrapper


class ProductAttributes(BaseModel):
    attributes: dict[str, str]


class RelationshipMetadata(BaseModel):
    metadata: dict[str, str]


class SubstituteProduct(BaseModel):
    product_id: str
    name: str
    price: float
    brand: str
    strength: float


class ComplementProduct(BaseModel):
    product_id: str
    name: str
    price: float
    brand: str
    strength: float
    relation_type: str


class ProductRecommendation(BaseModel):
    product_id: str
    name: str
    price: float
    brand: str
    relevance_score: float


class RecommendationContext(BaseModel):
    context: dict[str, str]


class QueryResults(BaseModel):
    rows: list[dict[str, str]]


class RetailKnowledgeGraph:
    def __init__(self, store_id: str, graph_uri: str | None = None):
        """Initialize the retail knowledge graph"""
        self.store_id = store_id
        # Initialize the RDF graph
        self.graph = Graph()
        # Define namespaces for our retail domain
        self.RETAIL = Namespace("http://retail.example.org/ontology#")
        self.PRODUCT = Namespace("http://retail.example.org/product/")
        self.CATEGORY = Namespace("http://retail.example.org/category/")
        self.STORE = Namespace("http://retail.example.org/store/")
        self.CUSTOMER = Namespace("http://retail.example.org/customer/")
        # endregion book:knowledge-graph-imports
        # region book:knowledge-graph-init-bindings
        # Bind namespaces to prefixes for easier querying
        self.graph.bind("retail", self.RETAIL)
        self.graph.bind("product", self.PRODUCT)
        self.graph.bind("category", self.CATEGORY)
        self.graph.bind("store", self.STORE)
        self.graph.bind("customer", self.CUSTOMER)
        # Load our retail ontology
        self._load_ontology()
        # Connect to external SPARQL endpoint if provided
        self.sparql_endpoint = None
        if graph_uri:
            self.sparql_endpoint = SPARQLWrapper(graph_uri)
            self.sparql_endpoint.setReturnFormat(JSON)
        # endregion book:knowledge-graph-init-bindings

    # region book:knowledge-graph-load-ontology-classes
    def _load_ontology(self):
        """Load the retail domain ontology into the graph"""
        # Define core classes
        self.graph.add((self.RETAIL.Product, RDF.type, RDFS.Class))
        self.graph.add((self.RETAIL.Category, RDF.type, RDFS.Class))
        self.graph.add((self.RETAIL.Store, RDF.type, RDFS.Class))
        self.graph.add((self.RETAIL.Customer, RDF.type, RDFS.Class))
        self.graph.add((self.RETAIL.Location, RDF.type, RDFS.Class))
        # endregion book:knowledge-graph-load-ontology-classes
        # region book:knowledge-graph-load-ontology-properties
        # Define properties
        self.graph.add((self.RETAIL.name, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.price, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.hasCategory, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.locatedIn, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.hasBrand, RDF.type, RDF.Property))
        # Define relationship properties
        # endregion book:knowledge-graph-load-ontology-properties
        # region book:knowledge-graph-load-ontology-relations
        self.graph.add((self.RETAIL.isSubstituteFor, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.complementsWith, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.isAccessoryFor, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.isVariantOf, RDF.type, RDF.Property))
        self.graph.add((self.RETAIL.purchased, RDF.type, RDF.Property))
        # endregion book:knowledge-graph-load-ontology-relations
        # region book:knowledge-graph-load-ontology-definitions
        # Add property definitions
        self.graph.add((self.RETAIL.isSubstituteFor, RDFS.domain, self.RETAIL.Product))
        self.graph.add((self.RETAIL.isSubstituteFor, RDFS.range, self.RETAIL.Product))
        self.graph.add((self.RETAIL.complementsWith, RDFS.domain, self.RETAIL.Product))
        self.graph.add((self.RETAIL.complementsWith, RDFS.range, self.RETAIL.Product))
        # Define symmetric properties
        self.graph.add((self.RETAIL.complementsWith, RDF.type, self.RETAIL.SymmetricProperty))
        # Define transitive properties
        self.graph.add((self.RETAIL.hasSubcategory, RDF.type, self.RETAIL.TransitiveProperty))
        # endregion book:knowledge-graph-load-ontology-definitions

    # region book:knowledge-graph-add-product
    def add_product(
        self,
        product_id: str,
        name: str,
        price: float,
        category_ids: list[str],
        brand: str,
        attributes: ProductAttributes,
    ) -> URIRef:
        """Add a product to the knowledge graph"""

        product_uri = self.PRODUCT[product_id]
        # Add basic product information
        self.graph.add((product_uri, RDF.type, self.RETAIL.Product))
        self.graph.add((product_uri, self.RETAIL.name, Literal(name)))
        self.graph.add((product_uri, self.RETAIL.price, Literal(price, datatype=XSD.decimal)))
        self.graph.add((product_uri, self.RETAIL.hasBrand, Literal(brand)))
        # endregion book:knowledge-graph-add-product
        # region book:knowledge-graph-add-product-categories
        # Add product categories
        for category_id in category_ids:
            category_uri = self.CATEGORY[category_id]
            self.graph.add((product_uri, self.RETAIL.hasCategory, category_uri))
        # Add product attributes
        for attr_name, attr_value in attributes.attributes.items():
            attr_property = self.RETAIL[attr_name]
            self.graph.add((product_uri, attr_property, Literal(attr_value)))

        return product_uri
        # endregion book:knowledge-graph-add-product-categories

    # region book:knowledge-graph-add-product-relationship
    def add_product_relationship(
        self,
        source_product_id: str,
        relationship_type: str,
        target_product_id: str,
        strength: float = 1.0,
        metadata: RelationshipMetadata | None = None,
    ):
        """Add a relationship between products"""
        source_uri = self.PRODUCT[source_product_id]
        target_uri = self.PRODUCT[target_product_id]
        # Map string relationship type to URI
        if relationship_type == "substitute":
            relation = self.RETAIL.isSubstituteFor
        elif relationship_type == "complement":
            relation = self.RETAIL.complementsWith
        elif relationship_type == "accessory":
            relation = self.RETAIL.isAccessoryFor
        elif relationship_type == "variant":
            relation = self.RETAIL.isVariantOf
        else:
            raise ValueError(f"Unknown relationship type: {relationship_type}")

        # Add the base relationship
        self.graph.add((source_uri, relation, target_uri))
        relation_node = None
        # Add strength as a reified statement
        if strength != 1.0:
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

        # Add any additional metadata
        if metadata:
            if relation_node is None:
                relation_node = BNode()
                self.graph.add((relation_node, RDF.type, RDF.Statement))
                self.graph.add((relation_node, RDF.subject, source_uri))
                self.graph.add((relation_node, RDF.predicate, relation))
                self.graph.add((relation_node, RDF.object, target_uri))
            for key, value in metadata.metadata.items():
                meta_property = self.RETAIL[key]

                self.graph.add((relation_node, meta_property, Literal(value)))

    # endregion book:knowledge-graph-add-product-relationship

    # region book:knowledge-graph-add-customer-purchase
    def add_customer_purchase(
        self,
        customer_id: str,
        product_id: str,
        timestamp: str,
        quantity: int = 1,
        order_id: str | None = None,
        channel: str | None = "in_store",
    ):
        """Record a customer purchase in the knowledge graph"""
        customer_uri = self.CUSTOMER[customer_id]
        product_uri = self.PRODUCT[product_id]
        # endregion book:knowledge-graph-add-customer-purchase

        # region book:knowledge-graph-add-customer-purchase-event
        # Create a purchase event
        purchase_node = BNode()
        self.graph.add((purchase_node, RDF.type, self.RETAIL.Purchase))
        self.graph.add((purchase_node, self.RETAIL.hasCustomer, customer_uri))
        self.graph.add((purchase_node, self.RETAIL.hasProduct, product_uri))
        self.graph.add(
            (
                purchase_node,
                self.RETAIL.timestamp,
                Literal(timestamp, datatype=XSD.dateTime),
            )
        )
        self.graph.add(
            (
                purchase_node,
                self.RETAIL.quantity,
                Literal(quantity, datatype=XSD.integer),
            )
        )
        # Add optional information
        if order_id:
            self.graph.add((purchase_node, self.RETAIL.orderID, Literal(order_id)))
        self.graph.add((purchase_node, self.RETAIL.channel, Literal(channel)))
        # Add direct customer-purchased-product relationship for convenience
        self.graph.add((customer_uri, self.RETAIL.purchased, product_uri))
        # endregion book:knowledge-graph-add-customer-purchase-event

    # region book:knowledge-graph-find-substitutes
    def find_substitutes(self, product_id: str, max_results: int = 5) -> list[SubstituteProduct]:
        """Find substitute products for a given product"""
        query = """
        PREFIX retail: <http://retail.example.org/ontology#>
        PREFIX product: <http://retail.example.org/product/>
        SELECT ?substitute ?name ?price ?brand ?strength
        WHERE {
            # Direct substitutes
            {
                product:__PRODUCT_ID__ retail:isSubstituteFor ?substitute .
                OPTIONAL {
                    ?stmt rdf:type rdf:Statement ;
                          rdf:subject product:__PRODUCT_ID__ ;
                          rdf:predicate retail:isSubstituteFor ;
                          rdf:object ?substitute ;
                              retail:strength ?strength .
                    }
                }

                # Reverse substitutes
                UNION
                {
                    ?substitute retail:isSubstituteFor product:__PRODUCT_ID__ .
                OPTIONAL {
                    ?stmt rdf:type rdf:Statement ;
                          rdf:subject ?substitute ;
                          rdf:predicate retail:isSubstituteFor ;
                          rdf:object product:__PRODUCT_ID__ ;
                              retail:strength ?strength .
                    }
                }

                # Category-based substitutes (same category, similar price)
                UNION
                {
                    product:__PRODUCT_ID__ retail:hasCategory ?category .
                ?substitute retail:hasCategory ?category .
                product:__PRODUCT_ID__ retail:price ?originalPrice .
                ?substitute retail:price ?price .
                # Only include products within 20%% of original price
                FILTER (?substitute != product:__PRODUCT_ID__)
                FILTER (?price >= ?originalPrice * 0.8 && ?price <= ?originalPrice * 1.2)
                # Use a default strength lower than explicit substitutes
                    BIND(0.7 as ?strength)
                }

                # Get additional properties
                ?substitute retail:name ?name .
                ?substitute retail:price ?price .
                ?substitute retail:hasBrand ?brand .
            # If no strength was specified, default to 1.0
            BIND(COALESCE(?strength, 1.0) as ?strength)
        }
        ORDER BY DESC(?strength) ?price
        LIMIT __LIMIT__
        """
        query = query.replace("__PRODUCT_ID__", product_id).replace("__LIMIT__", str(max_results))
        # endregion book:knowledge-graph-find-substitutes

        # region book:knowledge-graph-find-substitutes-results
        results = self._execute_query(query).rows
        substitutes = []
        for row in results:
            substitute_uri = row["substitute"]
            substitute_id = substitute_uri.split("/")[-1]
            substitutes.append(
                SubstituteProduct(
                    product_id=substitute_id,
                    name=row["name"],
                    price=float(row["price"]),
                    brand=row["brand"],
                    strength=float(row["strength"]),
                )
            )

        return substitutes
        # endregion book:knowledge-graph-find-substitutes-results

    # region book:knowledge-graph-find-complements
    def find_complementary_products(self, product_id: str, max_results: int = 5) -> list[ComplementProduct]:
        """Find products that complement a given product"""
        query = """
        PREFIX retail: <http://retail.example.org/ontology#>
        PREFIX product: <http://retail.example.org/product/>

        SELECT ?complement ?name ?price ?brand ?strength ?relation_type
        WHERE {
            # Direct complements
            {
                product:__PRODUCT_ID__ retail:complementsWith ?complement .
                BIND("complement" AS ?relation_type)
                OPTIONAL {
                    ?stmt rdf:type rdf:Statement ;
                          rdf:subject product:__PRODUCT_ID__ ;
                          rdf:predicate retail:complementsWith ;
                          rdf:object ?complement ;
                              retail:strength ?strength .
                    }
                }

                # Accessories
                UNION
                {
                    ?complement retail:isAccessoryFor product:__PRODUCT_ID__ .
                BIND("accessory" AS ?relation_type)
                OPTIONAL {
                    ?stmt rdf:type rdf:Statement ;
                          rdf:subject ?complement ;
                          rdf:predicate retail:isAccessoryFor ;
                          rdf:object product:__PRODUCT_ID__ ;
                              retail:strength ?strength .
                    }
                }

                # Frequently bought together (derived from purchase data)
                UNION
                {
                    SELECT ?complement (COUNT(*) as ?count) ("co_purchase" AS ?relation_type)
                WHERE {
                    ?purchase1 retail:hasProduct product:__PRODUCT_ID__ ;
                              retail:hasCustomer ?customer ;
                              retail:orderID ?order .
                    ?purchase2 retail:hasProduct ?complement ;
                              retail:hasCustomer ?customer ;
                              retail:orderID ?order .
                    FILTER(?complement != product:__PRODUCT_ID__)
                }
                GROUP BY ?complement
                    HAVING(COUNT(*) >= 5)  # Minimum co-purchase threshold
                }

                # Get additional properties
                ?complement retail:name ?name .
    # endregion book:knowledge-graph-find-complements
                ?complement retail:price ?price .
                ?complement retail:hasBrand ?brand .
            # Calculate strength for co-purchases, or use default
            BIND(
                IF(?relation_type = "co_purchase",
                   ?count / 20, # Normalize co-purchase count
                       COALESCE(?strength, 1.0))
                    AS ?strength
                )
            }

            ORDER BY DESC(?strength) ?relation_type
            LIMIT __LIMIT__
        """
        query = query.replace("__PRODUCT_ID__", product_id).replace("__LIMIT__", str(max_results))

        # region book:knowledge-graph-find-complements-results
        results = self._execute_query(query).rows
        complements = []
        for row in results:
            complement_uri = row["complement"]
            complement_id = complement_uri.split("/")[-1]
            complements.append(
                ComplementProduct(
                    product_id=complement_id,
                    name=row["name"],
                    price=float(row["price"]),
                    brand=row["brand"],
                    strength=float(row["strength"]),
                    relation_type=row["relation_type"],
                )
            )

        return complements
        # endregion book:knowledge-graph-find-complements-results

    # region book:knowledge-graph-execute-query
    def _execute_query(self, query_str: str) -> QueryResults:
        """Execute a SPARQL query against the knowledge graph"""
        rows: list[dict[str, str]] = []
        if self.sparql_endpoint:
            # Use external SPARQL endpoint
            self.sparql_endpoint.setQuery(query_str)
            results = self.sparql_endpoint.query().convert()
            for row in results["results"]["bindings"]:
                parsed: dict[str, str] = {}
                for key, value in row.items():
                    if isinstance(value, dict) and "value" in value:
                        parsed[key] = str(value["value"])
                    else:
                        parsed[key] = str(value)
                rows.append(parsed)
        else:
            # Use local graph
            qres = self.graph.query(query_str)
            for row in qres:
                result: dict[str, str] = {}
                for var in row.labels:
                    result[var] = str(row[var])
                rows.append(result)
        return QueryResults(rows=rows)

    # endregion book:knowledge-graph-execute-query

    # region book:knowledge-graph-generate-recommendations
    def generate_recommendations(
        self,
        customer_id: str,
        current_context: RecommendationContext | None = None,
        max_results: int = 5,
    ) -> list[ProductRecommendation]:
        """Generate personalized recommendations for a customer"""
        # Base query using purchase history
        query = """
        PREFIX retail: <http://retail.example.org/ontology#>
        PREFIX customer: <http://retail.example.org/customer/>
        SELECT DISTINCT ?product ?name ?price ?brand ?score
        WHERE {
            # Find products similar to what the customer has purchased
            {
                customer:%s retail:purchased ?purchasedProduct .
                ?purchasedProduct retail:hasCategory ?category .
                ?product retail:hasCategory ?category .
                # Avoid recommending products they already purchased
                FILTER(?product != ?purchasedProduct)
                # Basic category-based score
                BIND(0.5 AS ?baseScore)
                # Get additional properties
                ?product retail:name ?name .
                ?product retail:price ?price .
                    ?product retail:hasBrand ?brand .
                }

                # Boost score for complementary products
                OPTIONAL {
                    customer:%s retail:purchased ?otherProduct .
                    ?product retail:complementsWith ?otherProduct .
                BIND(0.3 AS ?complementBoost)
            }
            # Calculate total score
            BIND(COALESCE(?baseScore, 0) + COALESCE(?complementBoost, 0) AS ?score)
            }
            ORDER BY DESC(?score) ?name
            LIMIT %d

        """ % (customer_id, customer_id, max_results)  # noqa: UP031

        # Add context-specific filters if provided
        if current_context:
            # We could enhance this query with the customer's current in-store location,
            # shopping list items, or other contextual information.
            pass
        results = self._execute_query(query).rows
        recommendations = []
        for row in results:
            product_uri = row["product"]
            product_id = product_uri.split("/")[-1]
            recommendations.append(
                ProductRecommendation(
                    product_id=product_id,
                    name=row["name"],
                    price=float(row["price"]),
                    brand=row["brand"],
                    relevance_score=float(row["score"]),
                )
            )

        return recommendations

    # endregion book:knowledge-graph-generate-recommendations

    # region book:knowledge-graph-export-load
    def export_graph(self, format: str = "turtle") -> str:
        """Export the knowledge graph in the specified format"""
        return self.graph.serialize(format=format)

    def load_graph(self, data: str, format: str = "turtle"):
        """Load data into the knowledge graph"""
        self.graph.parse(data=data, format=format)

    # endregion book:knowledge-graph-export-load

    # region book:knowledge-graph-clear-graph
    def clear_graph(self):
        """Clear all data from the graph except the ontology"""
        # Store the ontology triples
        ontology_triples = [
            triple
            for triple in self.graph
            if triple[0].startswith(self.RETAIL) and triple[1] in (RDF.type, RDFS.domain, RDFS.range)
        ]
        # Clear the graph
        self.graph = Graph()
        # Restore namespaces
        self.graph.bind("retail", self.RETAIL)
        self.graph.bind("product", self.PRODUCT)
        self.graph.bind("category", self.CATEGORY)
        self.graph.bind("store", self.STORE)
        self.graph.bind("customer", self.CUSTOMER)
        # Restore ontology triples
        for triple in ontology_triples:
            self.graph.add(triple)

    # endregion book:knowledge-graph-clear-graph
