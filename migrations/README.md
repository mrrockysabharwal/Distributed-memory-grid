# Database migrations

Run commands from the repository root with `DATABASE_URL` set to the target
database and a valid `JWT_SECRET_KEY` available through the environment or
`.env`.

## Fresh database

Create the original `users` and `inference_jobs` tables and add job ownership:

```powershell
python -m alembic upgrade head
```

## Existing database without Alembic history

Back up the database first. Confirm `inference_jobs` has its original schema
and does not already have `user_id`. If `users` already exists, confirm its
schema and unique email index match the baseline. `stamp` only records a
revision; it does not inspect, create, or change tables. Stamp the database at
the baseline and then apply the ownership migration:

```powershell
python -m alembic stamp 20261126_0001
python -m alembic upgrade head
```

This procedure supports both existing schema states:

- If the database already has `users` and `inference_jobs`, the ownership
  migration leaves both tables and their records intact and adds only the
  nullable foreign key and index.
- If it has `inference_jobs` but no `users` table, the ownership migration
  creates `users` with the same columns, constraints, client defaults, and
  unique email index as the baseline migration, then adds the nullable foreign
  key. It does not recreate or overwrite `inference_jobs`.

In either case, existing jobs remain with `user_id = NULL`; no legacy job is
assigned to an arbitrary user. Do not stamp a database whose `inference_jobs`
schema differs from the original baseline or already contains an untracked
`user_id` column. Reconcile such a schema separately before upgrading.

## Downgrading

To undo only job ownership while preserving job rows, user rows, and the
original tables, downgrade to the baseline. If the ownership migration created
`users` because it was missing, that table and any user records added after
the upgrade are also retained:

```powershell
python -m alembic downgrade 20261126_0001
```

Downgrading to `base` also drops the baseline tables. That operation deletes
their data and should only be used for disposable databases.
