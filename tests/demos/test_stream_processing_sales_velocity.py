import re
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock


def test_book_stream_emits_kafka_key_and_value_columns(monkeypatch):
    # Fake client enforces the Kafka sink's column contract without Spark/JVM/Kafka.
    stream, aggregated, encoded = MagicMock(), MagicMock(), MagicMock()
    for method in ["format", "option", "load", "selectExpr", "select", "withWatermark", "groupBy"]:
        getattr(stream, method).return_value = stream
    stream.agg.return_value = aggregated
    aggregated.columns = ["product_id", "store_id", "window", "total_quantity"]

    def select_expressions(*expressions):
        encoded.columns = [
            re.search(r"\bAS\s+(\w+)$", expression, re.IGNORECASE)[1] for expression in expressions
        ]
        return encoded

    aggregated.selectExpr.side_effect = select_expressions

    for frame in [aggregated, encoded]:
        writer = frame.writeStream
        for method in ["outputMode", "format", "option"]:
            getattr(writer, method).return_value = writer

        def start(frame=frame):
            if "value" not in frame.columns:
                raise ValueError("Kafka sink requires a value column")
            return MagicMock()

        writer.start.side_effect = start

    sql, functions, types = (
        ModuleType("pyspark.sql"),
        ModuleType("pyspark.sql.functions"),
        ModuleType("pyspark.sql.types"),
    )
    sql.SparkSession = SimpleNamespace(builder=MagicMock())
    sql.SparkSession.builder.appName.return_value.getOrCreate.return_value.readStream = stream
    for name in ["avg", "col", "count", "from_json", "sum", "window"]:
        setattr(functions, name, MagicMock())
    for name in ["DoubleType", "StringType", "StructField", "StructType", "TimestampType"]:
        setattr(types, name, MagicMock())
    for name, module in [
        ("pyspark", ModuleType("pyspark")),
        ("pyspark.sql", sql),
        ("pyspark.sql.functions", functions),
        ("pyspark.sql.types", types),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[2] / "demos/stream_processing_sales_velocity_demo.py"
    runpy.run_path(str(path))
    assert set(encoded.columns) == {"key", "value"}
