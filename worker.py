"""
Worker para processar documentos da fila SQS.

Uso:
    python worker.py

O worker fica em loop consumindo mensagens da fila e processando documentos.
"""
import json
import asyncio
import logging
from uuid import UUID

from app.queue.sqs import receive_messages, delete_message
from app.rag.ingest import ingest_document
from app.db.session import AsyncSessionLocal
from app.core.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def process_message(message: dict) -> bool:
    """Processa uma mensagem da fila."""
    try:
        body = json.loads(message["Body"])
        document_id = UUID(body["document_id"])
        file_path = body["file_path"]

        logger.info(f"Processando documento {document_id}")

        async with AsyncSessionLocal() as db:
            await ingest_document(db, document_id, file_path)

        logger.info(f"Documento {document_id} processado com sucesso")
        return True

    except Exception as e:
        logger.error(f"Erro ao processar mensagem: {e}")
        return False


async def run_worker():
    """Loop principal do worker."""
    logger.info("Worker iniciado")
    logger.info(f"Consumindo fila: {settings.SQS_QUEUE_URL}")

    while True:
        try:
            messages = receive_messages(max_messages=5, wait_time=20)

            if not messages:
                continue

            for message in messages:
                success = await process_message(message)

                if success:
                    delete_message(message["ReceiptHandle"])
                    logger.info(f"Mensagem deletada: {message['MessageId']}")

        except KeyboardInterrupt:
            logger.info("Worker encerrado")
            break
        except Exception as e:
            logger.error(f"Erro no worker: {e}")
            await asyncio.sleep(5)


if __name__ == "__main__":
    if not settings.SQS_QUEUE_URL:
        print("Erro: SQS_QUEUE_URL nao configurado")
        exit(1)

    asyncio.run(run_worker())
