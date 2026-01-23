import json
import boto3
from uuid import UUID
from app.core.config import settings

_client = None

def get_sqs_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "sqs",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
    return _client


def send_ingest_message(document_id: UUID, file_path: str) -> str:
    """Envia mensagem para fila de processamento."""
    client = get_sqs_client()

    message = {
        "document_id": str(document_id),
        "file_path": file_path,
    }

    response = client.send_message(
        QueueUrl=settings.SQS_QUEUE_URL,
        MessageBody=json.dumps(message),
    )
    return response["MessageId"]


def receive_messages(max_messages: int = 10, wait_time: int = 20):
    """Recebe mensagens da fila."""
    client = get_sqs_client()

    response = client.receive_message(
        QueueUrl=settings.SQS_QUEUE_URL,
        MaxNumberOfMessages=max_messages,
        WaitTimeSeconds=wait_time,
    )
    return response.get("Messages", [])


def delete_message(receipt_handle: str) -> None:
    """Deleta mensagem da fila apos processamento."""
    client = get_sqs_client()
    client.delete_message(
        QueueUrl=settings.SQS_QUEUE_URL,
        ReceiptHandle=receipt_handle,
    )
