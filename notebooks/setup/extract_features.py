import argparse
import os
import time

import boto3

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--staging", required=True)
    parser.add_argument("--region", required=True)
    args = parser.parse_args()

    athena = boto3.client("athena", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    query = f"""
        SELECT * FROM (
            SELECT *, row_number() OVER (
                PARTITION BY encounter_id
                ORDER BY event_time DESC, api_invocation_time DESC
            ) AS rn
            FROM "{args.table}"
            WHERE NOT is_deleted
        ) WHERE rn = 1
    """

    qid = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": args.database},
        ResultConfiguration={"OutputLocation": args.staging},
    )["QueryExecutionId"]

    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        if status["State"] in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(3)

    if status["State"] != "SUCCEEDED":
        raise RuntimeError(f"Athena query {status['State']}: {status.get('StateChangeReason')}")

    result_uri = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["ResultConfiguration"]["OutputLocation"]
    bucket, key = result_uri.replace("s3://", "").split("/", 1)

    out_dir = "/opt/ml/processing/output"
    os.makedirs(out_dir, exist_ok=True)
    s3.download_file(bucket, key, f"{out_dir}/features.csv")
    print("Extracted Feature Store data to features.csv from", result_uri)
