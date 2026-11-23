"""Persistence operations for inference jobs."""

from sqlalchemy.orm import Session

from backend.models.inference_job import InferenceJob


def create_job(
    db: Session,
    *,
    job_id: str,
    model_id: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> InferenceJob:
    job = InferenceJob(
        job_id=job_id,
        model_id=model_id,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        status="queued",
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job(db: Session, job_id: str) -> InferenceJob | None:
    return db.get(InferenceJob, job_id)
