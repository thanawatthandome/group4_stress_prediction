from awsglue.context import GlueContext 
from pyspark.context import SparkContext
from pyspark.sql.functions import col, when
from pyspark.sql.types import FloatType

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

input_path = "s3://stress-prediction-project/rawdata_training/unclean_smartwatch_health_data.csv"
output_path = "s3://stress-prediction-project/preprocessed_data_training_xgboost/"

def to_float_safe(col_name):
    return when(col(col_name).rlike(r'^\d+(\.\d+)?$'), col(col_name).cast(FloatType())).otherwise(None)

try:
    df = spark.read.option("header", True).csv(input_path)
    print("Schema:")
    df.printSchema()

    numeric_cols = ["Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count", "Sleep Duration (hours)"]
    for nc in numeric_cols:
        df = df.withColumn(nc, to_float_safe(nc))

    df = df.withColumn("Stress Level", col("Stress Level").cast("int"))
    df = df.withColumn("User ID", col("User ID").cast("int"))

    df = df.withColumn(
         "Stress Level",
         when(col("Stress Level").between(1, 2), 0)
         .when(col("Stress Level").between(3, 4), 1)
         .when(col("Stress Level").between(5, 6), 2)
         .when(col("Stress Level").between(7, 8), 3)
         .when(col("Stress Level").between(9, 10), 4)
         .otherwise(None)
     )

    df = df.withColumn(
        "Activity Level",
        when(col("Step Count") < 5000, "Sedentary")
        .when((col("Step Count") >= 5000) & (col("Step Count") < 7500), "Lightly Active")
        .when((col("Step Count") >= 7500) & (col("Step Count") < 10000), "Moderately Active")
        .when((col("Step Count") >= 10000) & (col("Step Count") < 12500), "Very Active")
        .otherwise("Highly Active")
    )

    df = df.withColumn(
        "ActivityLevelIndex",
        when(col("Activity Level") == "Sedentary", 1)
        .when(col("Activity Level") == "Lightly Active", 2)
        .when(col("Activity Level") == "Moderately Active", 3)
        .when(col("Activity Level") == "Very Active", 4)
        .when(col("Activity Level") == "Highly Active", 5)
        .otherwise(None)
    )

    df = df.filter(
        (col("Heart Rate (BPM)").between(40, 150)) &
        (col("Blood Oxygen Level (%)").between(82, 100))
    )

    for col_name in numeric_cols:
        mean_val = df.select(col_name).dropna().groupBy().avg().first()[0]
        if mean_val is not None:
            df = df.na.fill({col_name: mean_val})

    median_stress = df.approxQuantile("Stress Level", [0.5], 0.01)[0]
    df = df.na.fill({"Stress Level": int(median_stress)})

    df = df.dropna(subset=[
        "User ID", "Heart Rate (BPM)", "Blood Oxygen Level (%)", "Step Count",
        "Sleep Duration (hours)", "ActivityLevelIndex", "Stress Level"
    ])

    # ✨ เลือก column โดยให้ "Stress Level" อยู่หน้าสุด
    output_columns = ["Stress Level", "User ID", "Heart Rate (BPM)", "Blood Oxygen Level (%)", 
                      "Step Count", "Sleep Duration (hours)", "ActivityLevelIndex"]
    df_output = df.select(output_columns)

    print("Saving CSV without header for XGBoost...")
    df_output.write.option("header", False).mode("overwrite").csv(output_path)

    print("✅ XGBoost-ready CSV saved to S3!")

except Exception as e:
    print("❌ ERROR occurred:", str(e))
    raise e

finally:
    spark.stop()
