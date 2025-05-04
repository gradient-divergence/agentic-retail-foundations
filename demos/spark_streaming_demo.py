"""
Demonstration of real-time sales velocity calculation using PySpark Streaming.

NOTE: This script requires a running Spark cluster configured with Kafka integration 
      and access to the specified Kafka brokers and topics.
Run using spark-submit.
"""

import logging

# Attempt to import PySpark modules - will fail if Spark is not installed
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import from_json, col, window, avg, sum, count
    from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    # Define dummy classes/functions if Spark not available, so file can be imported
    # without runtime error, although it won't function.
    class SparkSession:
        @staticmethod
        def builder(): return SparkSession()
        def appName(self, name): return self
        def getOrCreate(self): return self
        def readStream(self): raise RuntimeError("PySpark not available")
        def stop(self): pass
    
    def col(c): return c # Simple passthrough
    def window(c, w, s): return c # Simple passthrough
    def avg(c): return c 
    def sum(c): return c 
    def count(c): return c
    def from_json(c, s): return c 
    StructType = lambda fields: None
    StructField = lambda name, type, null: None
    StringType = lambda: None
    TimestampType = lambda: None
    DoubleType = lambda: None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("spark-streaming-demo")

def run_spark_streaming_job():
    """Runs the Spark Streaming job for sales velocity calculation.""" 
    if not SPARK_AVAILABLE:
        logger.error("PySpark is not installed or available. Cannot run Spark Streaming job.")
        print("Error: PySpark not found. Please install pyspark and ensure Spark environment is configured.")
        return

    logger.info("Initializing Spark Session...")
    spark = (
        SparkSession.builder
        .appName("RetailSalesVelocity")
        # Add Kafka package configuration if needed, e.g.:
        # .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.0") 
        .getOrCreate()
    )
    # Set log level for Spark
    spark.sparkContext.setLogLevel("WARN") 
    logger.info("Spark Session created.")

    # Schema for incoming sales data stream (from notebook)
    schema = StructType([
        StructField("product_id", StringType(), True),
        StructField("store_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("price", DoubleType(), True),
        StructField("quantity", DoubleType(), True),
        StructField("total_value", DoubleType(), True),
    ])
    logger.info("Defined input schema.")

    # Kafka configuration
    KAFKA_BROKERS = "localhost:9092" # Replace with actual Kafka brokers
    INPUT_TOPIC = "sales-transactions"
    OUTPUT_TOPIC = "sales-velocity-metrics"
    CHECKPOINT_LOCATION = "/tmp/spark-checkpoints/sales-velocity" # Use HDFS or reliable storage in production

    logger.info(f"Reading from Kafka topic: {INPUT_TOPIC} at {KAFKA_BROKERS}")
    # Read from Kafka stream
    sales_stream = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKERS)
        .option("subscribe", INPUT_TOPIC)
        .option("startingOffsets", "latest") # Process new messages
        .load()
        .selectExpr("CAST(value AS STRING)") # Kafka value is binary
        .select(from_json(col("value"), schema).alias("data")) # Parse JSON
        .select("data.*")
    )

    logger.info("Calculating rolling sales velocity (15 min window, 5 min slide)...")
    # Calculate rolling sales velocity over 15-minute windows, sliding every 5 mins
    sales_velocity = (
        sales_stream
        .withWatermark("timestamp", "10 minutes") # Handle late data
        .groupBy(
            col("product_id"),
            col("store_id"),
            window(col("timestamp"), "15 minutes", "5 minutes"),
        )
        .agg(
            avg("quantity").alias("avg_quantity_per_transaction"),
            sum("quantity").alias("total_quantity"),
            avg("price").alias("avg_price"),
            count("*").alias("transaction_count"),
        )
    )
    
    logger.info(f"Writing results to Kafka topic: {OUTPUT_TOPIC}")
    # Write the aggregated data to Kafka
    # Need to select key/value and serialize value to JSON string
    query = (
        sales_velocity.selectExpr(
             "CAST(product_id AS STRING) AS key", # Use product_id as key
             "to_json(struct(*)) AS value" # Convert row to JSON string
        )
        .writeStream
        .outputMode("update") # Use update mode for aggregations
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKERS)
        .option("topic", OUTPUT_TOPIC)
        .option("checkpointLocation", CHECKPOINT_LOCATION)
        .start()
    )

    logger.info("Spark Streaming job started. Waiting for termination...")
    # Keep the job running until manually stopped or an error occurs
    try:
         query.awaitTermination()
    except Exception as e:
         logger.error(f"Spark Streaming job failed: {e}", exc_info=True)
    finally:
         logger.info("Stopping Spark Session.")
         spark.stop()

if __name__ == "__main__":
    run_spark_streaming_job() 