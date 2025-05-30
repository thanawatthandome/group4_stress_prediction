import sys
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.utils import getResolvedOptions
from pyspark.sql.functions import col, udf, to_date, when
from pyspark.sql.types import IntegerType, FloatType

# ===== Step 1: Glue Setup =====
args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session

job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# ===== Step 2: Read JSON input from S3 =====
input_path = "s3://stress-prediction-project/rawdata_steaming/2025/"
df = spark.read.option("recursiveFileLookup", "true").json(input_path)

# Optional: Debug schema
df.printSchema()

# ===== Step 3: Map Activity Level to Number =====
def activity_to_num(level):
    if level == "Sedentary":
        return 1
    elif level == "Low active":
        return 2
    elif level == "Moderately active":
        return 3
    elif level == "Highly active":
        return 4
    elif level == "Very highly active":
        return 5
    else:
        return 0

map_udf = udf(activity_to_num, IntegerType())

# ===== Step 4: Preprocessing =====
numeric_cols = ["Sleep Duration", "Step Count", "blood oxygen level", "bpm"]
for nc in numeric_cols:
    df = df.withColumn(nc, when(col(nc).rlike(r'^\d+(\.\d+)?$'), col(nc).cast(FloatType())).otherwise(None))

df = df.withColumn("Activity_Level_Num", map_udf(col("Activity Level")))

df = df.filter(
    (col("bpm").between(40, 150)) &
    (col("blood oxygen level").between(82, 100)) &
    (col("Activity_Level_Num") > 0)
)

# Fill missing values with column mean
for nc in numeric_cols:
    mean_val = df.select(nc).dropna().groupBy().avg().first()[0]
    df = df.na.fill({nc: mean_val})

df = df.withColumn("date", to_date(col("timestamp")))

# ===== Step 5: Select columns for prediction =====
output_columns = ["UserId", "bpm", "blood oxygen level", "Step Count", "Sleep Duration", "Activity_Level_Num", "timestamp", "date"]
df_output = df.select(output_columns)

# ===== Step 6: Combine & Export to Single CSV =====
output_path = "s3://stress-prediction-project/combined_data_steaming/"

df_output.coalesce(1).write \
    .mode("overwrite") \
    .option("header", "true") \
    .csv(output_path)

# ===== Step 7: Finish =====
job.commit()
