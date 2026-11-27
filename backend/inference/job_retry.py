"""
Mid-Generation Job Retry and Checkpoint Recovery Engine.
Provides automatic retry logic for inference jobs that fail mid-generation
due to worker crashes, OOM errors, or transient network failures.

Features:
- Periodic generation checkpoint snapshots (token IDs + KV cache state hashes).
- Configurable retry policy with exponential backoff and max attempts.
- Checkpoint-based resume: restarts generation from the last good checkpoint
  rather than re-running the entire prefill from scratch.
- Dead-letter queue for jobs that exhaust all retries.
- Integration with KVCacheManager for cache state preservation.

Maintained by the Distributed Memory Grid team.
"""
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger("DMG-JobRetry")


class RetryReason(Enum):
    """Categorized failure reasons triggering a retry."""
    WORKER_CRASH = "worker_crash"
    OOM_ERROR = "oom_error"
    TIMEOUT = "timeout"
    NETWORK_ERROR = "network_error"
    UNKNOWN = "unknown"


class JobState(Enum):
    """State machine for a retryable job."""
    RUNNING = "running"
    CHECKPOINTED = "checkpointed"
    FAILED = "failed"
    RETRYING = "retrying"
    COMPLETED = "completed"
    DEAD_LETTERED = "dead_lettered"


@dataclass
class GenerationCheckpoint:
    """Snapshot of generation progress at a specific token position."""
    request_id: str
    token_position: int
    generated_token_ids: List[int]
    created_at: float = field(default_factory=time.time)
    kv_cache_layers: Optional[List[int]] = None  # Layer indices with cached state
    prompt_tokens: int = 0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at


@dataclass
class RetryRecord:
    """Tracks retry attempts for a single job."""
    request_id: str
    attempt: int = 0
    reason: RetryReason = RetryReason.UNKNOWN
    failed_at: float = field(default_factory=time.time)
    next_retry_at: float = 0.0
    error_message: str = ""


@dataclass
class RetryPolicy:
    """Configurable retry behavior."""
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0
    checkpoint_interval_tokens: int = 10
    retry_on: Optional[List[RetryReason]] = None

    def __post_init__(self):
        if self.retry_on is None:
            self.retry_on = [
                RetryReason.WORKER_CRASH,
                RetryReason.OOM_ERROR,
                RetryReason.TIMEOUT,
                RetryReason.NETWORK_ERROR,
            ]

    def compute_delay(self, attempt: int) -> float:
        """Compute exponential backoff delay for the given attempt number."""
        delay = self.base_delay_seconds * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay_seconds)

    def should_retry(self, reason: RetryReason, attempt: int) -> bool:
        """Determine if the job should be retried given the failure reason and attempt."""
        if attempt >= self.max_retries:
            return False
        return reason in self.retry_on


class JobRetryManager:
    """
    Manages mid-generation retry and checkpoint recovery for inference jobs.

    Integrates with the generation engine to:
    1. Periodically checkpoint generation progress.
    2. Detect failures and determine retry eligibility.
    3. Resume from the last checkpoint on retry (skipping completed prefill).
    4. Move exhausted jobs to a dead-letter queue for manual inspection.
    """

    def __init__(
        self,
        policy: Optional[RetryPolicy] = None,
        on_retry_callback: Optional[Callable[[str, int, "GenerationCheckpoint"], None]] = None,
    ):
        """
        Args:
            policy: Retry policy configuration.
            on_retry_callback: Called when a job is retried, with (request_id, attempt, checkpoint).
        """
        self.policy = policy or RetryPolicy()
        self.on_retry_callback = on_retry_callback

        # Active job state tracking
        self._job_states: Dict[str, JobState] = {}
        self._checkpoints: Dict[str, GenerationCheckpoint] = {}
        self._retry_records: Dict[str, List[RetryRecord]] = {}
        self._dead_letter_queue: List[Dict[str, Any]] = []

        # Statistics
        self._total_retries = 0
        self._total_recoveries = 0
        self._total_dead_lettered = 0

    def register_job(self, request_id: str, prompt_tokens: int = 0):
        """Register a new job for retry tracking."""
        self._job_states[request_id] = JobState.RUNNING
        self._retry_records[request_id] = []
        logger.debug(f"Job '{request_id}' registered for retry tracking.")

    def checkpoint(
        self,
        request_id: str,
        token_position: int,
        generated_token_ids: List[int],
        kv_cache_layers: Optional[List[int]] = None,
        prompt_tokens: int = 0,
    ) -> GenerationCheckpoint:
        """
        Save a generation checkpoint for the given job.

        Should be called periodically (every `checkpoint_interval_tokens` tokens).
        """
        cp = GenerationCheckpoint(
            request_id=request_id,
            token_position=token_position,
            generated_token_ids=list(generated_token_ids),  # defensive copy
            kv_cache_layers=kv_cache_layers,
            prompt_tokens=prompt_tokens,
        )
        self._checkpoints[request_id] = cp
        self._job_states[request_id] = JobState.CHECKPOINTED
        logger.debug(
            f"Checkpoint saved for '{request_id}' at position {token_position} "
            f"({len(generated_token_ids)} tokens generated)."
        )
        return cp

    def should_checkpoint(self, request_id: str, current_position: int) -> bool:
        """Check if a checkpoint should be taken at the current position."""
        if current_position == 0:
            return False
        return current_position % self.policy.checkpoint_interval_tokens == 0

    def report_failure(
        self,
        request_id: str,
        reason: RetryReason = RetryReason.UNKNOWN,
        error_message: str = "",
    ) -> Dict[str, Any]:
        """
        Report a job failure and determine whether to retry or dead-letter.

        Returns:
            Dict with keys:
            - should_retry (bool): Whether the job should be retried.
            - attempt (int): Current attempt number.
            - delay_seconds (float): Backoff delay before retry.
            - checkpoint (GenerationCheckpoint or None): Checkpoint to resume from.
            - dead_lettered (bool): Whether the job was moved to dead-letter queue.
        """
        self._job_states[request_id] = JobState.FAILED
        records = self._retry_records.get(request_id, [])
        attempt = len(records)

        record = RetryRecord(
            request_id=request_id,
            attempt=attempt,
            reason=reason,
            failed_at=time.time(),
            error_message=error_message,
        )

        if self.policy.should_retry(reason, attempt):
            delay = self.policy.compute_delay(attempt)
            record.next_retry_at = time.time() + delay
            records.append(record)
            self._retry_records[request_id] = records
            self._job_states[request_id] = JobState.RETRYING
            self._total_retries += 1

            checkpoint = self._checkpoints.get(request_id)
            logger.info(
                f"Job '{request_id}' failed ({reason.value}), scheduling retry "
                f"#{attempt + 1}/{self.policy.max_retries} in {delay:.1f}s"
                + (f" from checkpoint at token {checkpoint.token_position}"
                   if checkpoint else " from scratch")
            )

            if self.on_retry_callback and checkpoint:
                self.on_retry_callback(request_id, attempt + 1, checkpoint)

            return {
                "should_retry": True,
                "attempt": attempt + 1,
                "delay_seconds": delay,
                "checkpoint": checkpoint,
                "dead_lettered": False,
            }
        else:
            # Exhausted retries → dead-letter
            records.append(record)
            self._retry_records[request_id] = records
            self._job_states[request_id] = JobState.DEAD_LETTERED
            self._total_dead_lettered += 1

            self._dead_letter_queue.append({
                "request_id": request_id,
                "total_attempts": attempt + 1,
                "last_reason": reason.value,
                "last_error": error_message,
                "dead_lettered_at": time.time(),
                "checkpoint": self._checkpoints.get(request_id),
            })

            logger.warning(
                f"Job '{request_id}' moved to dead-letter queue after "
                f"{attempt + 1} attempts (last reason: {reason.value})."
            )
            return {
                "should_retry": False,
                "attempt": attempt + 1,
                "delay_seconds": 0.0,
                "checkpoint": self._checkpoints.get(request_id),
                "dead_lettered": True,
            }

    def get_resume_state(self, request_id: str) -> Dict[str, Any]:
        """
        Get the state needed to resume a job from its last checkpoint.

        Returns:
            Dict with:
            - has_checkpoint (bool)
            - resume_from_token (int): Token position to resume from.
            - generated_so_far (List[int]): Tokens generated before failure.
            - attempt (int): Current retry attempt number.
        """
        checkpoint = self._checkpoints.get(request_id)
        records = self._retry_records.get(request_id, [])

        if checkpoint is None:
            return {
                "has_checkpoint": False,
                "resume_from_token": 0,
                "generated_so_far": [],
                "attempt": len(records),
            }

        return {
            "has_checkpoint": True,
            "resume_from_token": checkpoint.token_position,
            "generated_so_far": list(checkpoint.generated_token_ids),
            "attempt": len(records),
        }

    def mark_completed(self, request_id: str):
        """Mark a job as successfully completed and clean up state."""
        was_retrying = request_id in self._retry_records and len(
            self._retry_records.get(request_id, [])
        ) > 0

        if was_retrying:
            self._total_recoveries += 1
            logger.info(f"Job '{request_id}' completed after retry recovery.")

        self._job_states[request_id] = JobState.COMPLETED
        # Clean up checkpoints (keep retry records for stats)
        self._checkpoints.pop(request_id, None)

    def cleanup_job(self, request_id: str):
        """Remove all tracking state for a job."""
        self._job_states.pop(request_id, None)
        self._checkpoints.pop(request_id, None)
        self._retry_records.pop(request_id, None)

    def get_dead_letter_queue(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent dead-lettered jobs."""
        return list(reversed(self._dead_letter_queue))[:limit]

    def retry_dead_letter(self, request_id: str) -> Optional[Dict[str, Any]]:
        """
        Manually retry a dead-lettered job (admin intervention).
        Resets the retry counter and returns resume state.
        """
        # Find and remove from dead-letter queue
        found = None
        for i, item in enumerate(self._dead_letter_queue):
            if item["request_id"] == request_id:
                found = self._dead_letter_queue.pop(i)
                break

        if found is None:
            return None

        # Reset state
        self._retry_records[request_id] = []
        self._job_states[request_id] = JobState.RETRYING
        self._total_dead_lettered -= 1

        return self.get_resume_state(request_id)

    def get_job_state(self, request_id: str) -> Optional[str]:
        """Get current state of a tracked job."""
        state = self._job_states.get(request_id)
        return state.value if state else None

    def stats(self) -> Dict[str, Any]:
        """Return retry system statistics."""
        state_counts = {}
        for state in self._job_states.values():
            state_counts[state.value] = state_counts.get(state.value, 0) + 1

        return {
            "total_tracked_jobs": len(self._job_states),
            "total_retries": self._total_retries,
            "total_recoveries": self._total_recoveries,
            "total_dead_lettered": self._total_dead_lettered,
            "active_checkpoints": len(self._checkpoints),
            "dead_letter_queue_size": len(self._dead_letter_queue),
            "job_state_counts": state_counts,
            "policy": {
                "max_retries": self.policy.max_retries,
                "base_delay_seconds": self.policy.base_delay_seconds,
                "checkpoint_interval_tokens": self.policy.checkpoint_interval_tokens,
            },
        }
