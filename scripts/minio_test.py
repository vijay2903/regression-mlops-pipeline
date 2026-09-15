from pathlib import Path

import boto3
from botocore.exceptions import ClientError


ENDPOINT_URL = "http://localhost:9000"
ACCESS_KEY = "minioadmin"
SECRET_KEY = "minioadmin"
BUCKET_NAME = "mlops-artifacts"

TEST_FILE = Path("results/minio-test.txt")
DOWNLOADED_FILE = Path("results/minio-test-downloaded.txt")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        region_name="us-east-1",
    )


def main():
    s3 = get_s3_client()

    print("Checking MinIO connection...")

    buckets = s3.list_buckets()["Buckets"]
    bucket_names = [bucket["Name"] for bucket in buckets]

    print("Available buckets:", bucket_names)

    if BUCKET_NAME not in bucket_names:
        print(f"Bucket '{BUCKET_NAME}' does not exist.")
        return

    TEST_FILE.parent.mkdir(exist_ok=True)
    TEST_FILE.write_text(
        "Hello from the California Housing MLOps project!",
        encoding="utf-8",
    )

    object_key = "tests/minio-test.txt"

    print("Uploading test file...")
    s3.upload_file(
        str(TEST_FILE),
        BUCKET_NAME,
        object_key,
    )

    print("Listing objects...")
    response = s3.list_objects_v2(Bucket=BUCKET_NAME)

    for obj in response.get("Contents", []):
        print(" -", obj["Key"])

    print("Downloading test file...")
    s3.download_file(
        BUCKET_NAME,
        object_key,
        str(DOWNLOADED_FILE),
    )

    original_content = TEST_FILE.read_text(encoding="utf-8")
    downloaded_content = DOWNLOADED_FILE.read_text(encoding="utf-8")

    if original_content == downloaded_content:
        print("SUCCESS: Upload and download verified.")
    else:
        print("ERROR: Downloaded content does not match.")


if __name__ == "__main__":
    main()