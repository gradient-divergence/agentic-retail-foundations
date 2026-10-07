from __future__ import annotations

# region book:spark-streaming-imports
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import avg, col, count, from_json, sum, window
    from pyspark.sql.types import (
        DoubleType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )
except ModuleNotFoundError:
    raise SystemExit("Install the spark extra: uv sync --extra spark") from None

schema = StructType(
    [
        StructField("product_id", StringType(), True),
        StructField("store_id", StringType(), True),
        StructField("timestamp", TimestampType(), True),
        StructField("price", DoubleType(), True),
        StructField("quantity", DoubleType(), True),
        StructField("total_value", DoubleType(), True),
    ]
)
# endregion book:spark-streaming-imports

if __name__ == "__main__":
    import socket

    try:
        with socket.create_connection(("kafka", 9092), timeout=1):
            pass
    except OSError:
        raise SystemExit("Start Kafka at kafka:9092 and a Spark JVM to run this demo.") from None


# region book:spark-streaming-session
spark = SparkSession.builder.appName("RetailStreamProcessor").getOrCreate()

sales_stream = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "sales-transactions")
    .load()
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)
# endregion book:spark-streaming-session


# region book:spark-streaming-velocity
sales_velocity = (
    sales_stream.withWatermark("timestamp", "1 minute")
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
# endregion book:spark-streaming-velocity


# region book:spark-streaming-query
query = (
    sales_velocity.selectExpr(
        "CAST(product_id AS STRING) AS key",
        "to_json(struct(*)) AS value",
    )
    .writeStream.outputMode("append")
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("topic", "sales-velocity-metrics")
    .option("checkpointLocation", "/checkpoints/sales-velocity")
    .start()
)

query.awaitTermination()
# endregion book:spark-streaming-query
