import os
import glob
import pandas as pd
import boto3

# ===== Step 1: Load CSV from Spark output folder =====
input_dir = "/opt/ml/processing/input/data/"

# 🧠 Spark export → part-0000*.csv
all_files = glob.glob(os.path.join(input_dir, "**"), recursive=True)
csv_candidates = [f for f in all_files if os.path.basename(f).startswith("part-") and f.endswith(".csv")]

if not csv_candidates:
    raise FileNotFoundError("❌ ไม่พบไฟล์ Spark .csv ที่ขึ้นต้นด้วย 'part-'")

csv_path = csv_candidates[0]
df = pd.read_csv(csv_path)
print(f"✅ Loaded Spark CSV: {csv_path} | Rows: {len(df)}")

# ===== Step 2: Prepare features for prediction =====
features = df[["bpm", "blood oxygen level", "Step Count", "Sleep Duration", "Activity_Level_Num"]]
print("🧠 Predicting on columns:", list(features.columns))

# ===== Step 3: Setup SageMaker Endpoint Runtime =====
runtime = boto3.client('sagemaker-runtime', region_name='us-east-1')
endpoint_name = 'xgboost-endpoint'

# ===== Step 4: Run prediction for each row =====
preds = []

for index, row in features.iterrows():
    payload = ','.join(map(str, row.values))
    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="text/csv",
        Body=payload
    )
    prediction = response['Body'].read().decode('utf-8').strip()
    preds.append(int(float(prediction)))

# ===== Step 5: Save prediction result =====
df["Predicted"] = preds
output_path = "/opt/ml/processing/output/predicted.csv"
df.to_csv(output_path, index=False)

print(f"✅ Prediction completed. Output saved to: {output_path}")
