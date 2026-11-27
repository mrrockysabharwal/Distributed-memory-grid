"""
Unit tests for Mid-Generation Job Retry and Checkpoint Recovery.
Tests cover checkpoint lifecycle, retry policy with exponential backoff,
dead-letter queue, manual retry, and resume state computation.
"""
from backend.inference.job_retry import (
    JobRetryManager,
    RetryPolicy,
    RetryReason,
    JobState,
    GenerationCheckpoint,
)


def test_register_and_checkpoint_job():
    """Registering and checkpointing a job stores the correct state."""
    mgr = JobRetryManager()
    mgr.register_job("req-1", prompt_tokens=10)
    assert mgr.get_job_state("req-1") == "running"

    cp = mgr.checkpoint("req-1", token_position=5, generated_token_ids=[100, 101, 102, 103, 104])
    assert cp.request_id == "req-1"
    assert cp.token_position == 5
    assert len(cp.generated_token_ids) == 5
    assert mgr.get_job_state("req-1") == "checkpointed"


def test_should_checkpoint_at_interval():
    """Checkpoints are requested at the configured interval."""
    policy = RetryPolicy(checkpoint_interval_tokens=10)
    mgr = JobRetryManager(policy=policy)
    mgr.register_job("req-1")

    assert mgr.should_checkpoint("req-1", 0) is False
    assert mgr.should_checkpoint("req-1", 5) is False
    assert mgr.should_checkpoint("req-1", 10) is True
    assert mgr.should_checkpoint("req-1", 20) is True
    assert mgr.should_checkpoint("req-1", 15) is False


def test_retry_on_worker_crash():
    """Worker crash triggers retry with backoff and checkpoint recovery."""
    mgr = JobRetryManager(policy=RetryPolicy(max_retries=3, base_delay_seconds=1.0))
    mgr.register_job("req-1")
    mgr.checkpoint("req-1", token_position=8, generated_token_ids=[10, 11, 12, 13, 14, 15, 16, 17])

    result = mgr.report_failure("req-1", reason=RetryReason.WORKER_CRASH, error_message="Worker died")
    assert result["should_retry"] is True
    assert result["attempt"] == 1
    assert result["delay_seconds"] == 1.0  # base_delay * 2^0
    assert result["checkpoint"] is not None
    assert result["checkpoint"].token_position == 8
    assert result["dead_lettered"] is False
    assert mgr.get_job_state("req-1") == "retrying"


def test_exponential_backoff():
    """Retry delays increase exponentially with each attempt."""
    policy = RetryPolicy(max_retries=5, base_delay_seconds=1.0, backoff_multiplier=2.0, max_delay_seconds=30.0)
    mgr = JobRetryManager(policy=policy)
    mgr.register_job("req-1")

    # Attempt 1
    r1 = mgr.report_failure("req-1", reason=RetryReason.TIMEOUT)
    assert r1["delay_seconds"] == 1.0  # 1 * 2^0

    # Attempt 2
    r2 = mgr.report_failure("req-1", reason=RetryReason.TIMEOUT)
    assert r2["delay_seconds"] == 2.0  # 1 * 2^1

    # Attempt 3
    r3 = mgr.report_failure("req-1", reason=RetryReason.TIMEOUT)
    assert r3["delay_seconds"] == 4.0  # 1 * 2^2


def test_max_delay_cap():
    """Backoff delay is capped at max_delay_seconds."""
    policy = RetryPolicy(max_retries=10, base_delay_seconds=10.0, backoff_multiplier=3.0, max_delay_seconds=30.0)
    assert policy.compute_delay(0) == 10.0
    assert policy.compute_delay(1) == 30.0  # 10 * 3 = 30
    assert policy.compute_delay(2) == 30.0  # 10 * 9 = 90 → capped at 30


def test_dead_letter_after_max_retries():
    """Jobs are dead-lettered when max retries are exhausted."""
    mgr = JobRetryManager(policy=RetryPolicy(max_retries=2))
    mgr.register_job("req-1")

    # Exhaust retries
    mgr.report_failure("req-1", reason=RetryReason.OOM_ERROR)
    mgr.report_failure("req-1", reason=RetryReason.OOM_ERROR)

    # 3rd failure should dead-letter
    r = mgr.report_failure("req-1", reason=RetryReason.OOM_ERROR)
    assert r["should_retry"] is False
    assert r["dead_lettered"] is True
    assert mgr.get_job_state("req-1") == "dead_lettered"

    dlq = mgr.get_dead_letter_queue()
    assert len(dlq) == 1
    assert dlq[0]["request_id"] == "req-1"


def test_manual_retry_dead_letter():
    """Admin can manually retry a dead-lettered job."""
    mgr = JobRetryManager(policy=RetryPolicy(max_retries=1))
    mgr.register_job("req-dl")
    mgr.checkpoint("req-dl", token_position=5, generated_token_ids=[1, 2, 3, 4, 5])

    # Exhaust retries
    mgr.report_failure("req-dl", reason=RetryReason.NETWORK_ERROR)
    mgr.report_failure("req-dl", reason=RetryReason.NETWORK_ERROR)

    assert mgr.get_job_state("req-dl") == "dead_lettered"

    # Manual retry
    resume = mgr.retry_dead_letter("req-dl")
    assert resume is not None
    assert resume["has_checkpoint"] is True
    assert resume["resume_from_token"] == 5
    assert mgr.get_job_state("req-dl") == "retrying"


def test_resume_state_with_checkpoint():
    """Resume state correctly returns checkpoint information."""
    mgr = JobRetryManager()
    mgr.register_job("req-resume")
    mgr.checkpoint("req-resume", token_position=12, generated_token_ids=list(range(12)))

    state = mgr.get_resume_state("req-resume")
    assert state["has_checkpoint"] is True
    assert state["resume_from_token"] == 12
    assert len(state["generated_so_far"]) == 12


def test_resume_state_without_checkpoint():
    """Resume state without a checkpoint starts from scratch."""
    mgr = JobRetryManager()
    mgr.register_job("req-fresh")

    state = mgr.get_resume_state("req-fresh")
    assert state["has_checkpoint"] is False
    assert state["resume_from_token"] == 0
    assert len(state["generated_so_far"]) == 0


def test_mark_completed_after_retry():
    """Completed jobs after retry are counted as recoveries."""
    mgr = JobRetryManager(policy=RetryPolicy(max_retries=3))
    mgr.register_job("req-recover")
    mgr.report_failure("req-recover", reason=RetryReason.WORKER_CRASH)
    mgr.mark_completed("req-recover")

    assert mgr.get_job_state("req-recover") == "completed"
    stats = mgr.stats()
    assert stats["total_recoveries"] == 1


def test_retry_reason_filtering():
    """RetryPolicy respects retry_on reason whitelist."""
    policy = RetryPolicy(
        max_retries=3,
        retry_on=[RetryReason.WORKER_CRASH, RetryReason.TIMEOUT],
    )
    # Allowed reasons
    assert policy.should_retry(RetryReason.WORKER_CRASH, attempt=0) is True
    assert policy.should_retry(RetryReason.TIMEOUT, attempt=0) is True
    # Not in whitelist
    assert policy.should_retry(RetryReason.OOM_ERROR, attempt=0) is False
    assert policy.should_retry(RetryReason.NETWORK_ERROR, attempt=0) is False


def test_cleanup_job_removes_all_state():
    """cleanup_job removes all tracking data."""
    mgr = JobRetryManager()
    mgr.register_job("req-cleanup")
    mgr.checkpoint("req-cleanup", 5, [1, 2, 3, 4, 5])
    mgr.report_failure("req-cleanup", reason=RetryReason.UNKNOWN)

    mgr.cleanup_job("req-cleanup")
    assert mgr.get_job_state("req-cleanup") is None
    assert mgr.get_resume_state("req-cleanup")["has_checkpoint"] is False


def test_retry_stats():
    """Stats reflect correct counts for all retry operations."""
    mgr = JobRetryManager(policy=RetryPolicy(max_retries=1))
    mgr.register_job("j1")
    mgr.register_job("j2")

    mgr.report_failure("j1", reason=RetryReason.TIMEOUT)
    mgr.mark_completed("j1")

    mgr.report_failure("j2", reason=RetryReason.OOM_ERROR)
    mgr.report_failure("j2", reason=RetryReason.OOM_ERROR)  # Dead-lettered

    stats = mgr.stats()
    assert stats["total_retries"] == 2  # j1 retried once, j2 retried once before dead-letter
    assert stats["total_recoveries"] == 1
    assert stats["total_dead_lettered"] == 1
    assert stats["total_tracked_jobs"] == 2
