"""Retry policies para tools."""
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.rag.agent_config import (
    TOOL_RETRY_MAX_ATTEMPTS,
    TOOL_RETRY_MIN_WAIT,
    TOOL_RETRY_MAX_WAIT,
)

logger = logging.getLogger(__name__)


# Exceções que NÃO devem fazer retry (erros "permanentes")
class PermanentToolError(Exception):
    """Erros que não resolvem com retry (ex: validação, tool desconhecida)."""
    pass


def tool_retry(func):
    """
    Decorator para retry em chamadas de tools.
    - Retry apenas em erros transientes (timeout, rede, rate limit)
    - NÃO faz retry em PermanentToolError
    - Backoff exponencial: 1s, 2s, 4s...
    """
    decorator = retry(
        stop=stop_after_attempt(TOOL_RETRY_MAX_ATTEMPTS),
        wait=wait_exponential(
            multiplier=1,
            min=TOOL_RETRY_MIN_WAIT,
            max=TOOL_RETRY_MAX_WAIT,
        ),
        retry=retry_if_exception_type((
            TimeoutError,
            ConnectionError,
            # adicionar outros erros transientes conforme necessário
        )),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    return decorator(func)
