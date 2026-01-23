import boto3
from uuid import UUID
from app.core.config import settings

_client = None

def get_s3_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
    return _client


def upload_to_s3(document_id: UUID, filename: str, content: bytes) -> str:
    """Upload arquivo para S3 e retorna a key."""
    client = get_s3_client()
    safe_name = filename.replace("/", "_").replace("\\", "_")
    key = f"documents/{document_id}_{safe_name}"

    client.put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=content,
    )
    return key


def download_from_s3(key: str) -> bytes:
    """Download arquivo do S3."""
    client = get_s3_client()
    response = client.get_object(Bucket=settings.S3_BUCKET, Key=key)
    return response["Body"].read()


def delete_from_s3(key: str) -> None:
    """Deleta arquivo do S3."""
    client = get_s3_client()
    client.delete_object(Bucket=settings.S3_BUCKET, Key=key)
