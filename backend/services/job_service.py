"""Persistence operations for inference jobs."""

from sqlalchemy.orm import Session

from backend.models.inference_job import InferenceJob
from backend.models.user import User


def create_job(
    db: Session,
    *,
    job_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    user: User,
) -> InferenceJob:
    if user.id is None:
        raise ValueError("Inference job owner must be a persisted user")

    job = InferenceJob(
        job_id=job_id,
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        user_id=user.id,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(
    db: Session,
    job_id: str,
    owner_user_id: int,
) -> InferenceJob | None:
    return (
        db.query(InferenceJob)
        .filter(
            InferenceJob.job_id == job_id,
            InferenceJob.user_id == owner_user_id,
        )
        .first()
    )
