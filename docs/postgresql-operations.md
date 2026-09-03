# PostgreSQL operations

MerchantShield uses PostgreSQL only for operational application state. IEEE-CIS and Online Retail II
raw files remain under `data/raw/`; processed ML tables remain Parquet under `data/processed/`.
Neither training dataset belongs in PostgreSQL.

## Local Docker database

Docker Desktop is required on macOS. Verify it before starting MerchantShield:

```bash
docker --version
docker compose version
```

Copy `.env.example` to `.env`, replace `change_me`, then start only PostgreSQL:

```bash
make db-up
make db-status
make db-logs
```

The API runs on the Mac with `localhost` in `DATABASE_URL`. Inside Compose, the API receives a URL
whose host is `postgres`, the Docker service name. The application code does not change between
those environments.

Open PostgreSQL directly:

```bash
make db-shell
```

Then verify:

```sql
SELECT current_database();
SELECT version();
\dt
SELECT * FROM alembic_version;
```

Apply and inspect migrations from the repository root:

```bash
make db-migrate
PYTHONPATH=ml/src:services/api .venv/bin/alembic -c services/api/alembic.ini current
```

Start the migrated API and verify both health paths:

```bash
make api
curl http://localhost:8000/health
curl http://localhost:8000/health/db
```

`/health` proves the process responds. `/health/db` executes `SELECT 1` through SQLAlchemy and proves
the configured database is reachable.

## Persistence proof

Use Review Queue to submit a decision and reviewer note, then inspect the stored `review_cases` row.
The current normalized record stores the decision, reason, reviewer identity, and timestamp.

```sql
SELECT rc.id, rc.status, rc.reviewer_decision, rc.reviewer_reason, rc.reviewed_at
FROM review_cases AS rc
WHERE rc.reviewed_at IS NOT NULL
ORDER BY rc.reviewed_at DESC
LIMIT 10;
```

Restart without deleting the named volume:

```bash
docker compose down
docker compose up -d postgres
```

Re-run the query to prove the decision survived. Do not use `docker compose down -v` unless deleting
the local database is intentional; `-v` removes `merchantshield_postgres_data`.

## Full local stack

```bash
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health
curl http://localhost:8000/health/db
```

The local API container applies `alembic upgrade head` before starting one Uvicorn process. Do not
copy this migration-on-start pattern to multiple production replicas because they could race.

To apply migrations explicitly inside the running API container:

```bash
docker compose exec api alembic -c services/api/alembic.ini upgrade head
docker compose exec api alembic -c services/api/alembic.ini current
```

## Intentional local reset

Ordinary restarts must preserve `merchantshield_postgres_data`. To intentionally erase the local
database and rebuild it from migrations, first confirm no local decisions or cases are needed, then
run:

```bash
docker compose down -v
docker compose up -d postgres
make db-migrate
```

The `-v` command is destructive and is not part of the persistence test.

## Hosted database

Use a managed PostgreSQL provider such as Neon or Supabase for a deployed FastAPI service. Set its
connection string only as the backend `DATABASE_URL`, including the provider's required TLS options.
Do not create `NEXT_PUBLIC_DATABASE_URL`; `NEXT_PUBLIC_` values are delivered to browser code.

Production startup rejects URLs that do not include `sslmode=require`, `sslmode=verify-ca`, or
`sslmode=verify-full`. Run the same Alembic history against Neon before starting the deployed API:

```bash
ENVIRONMENT=production \
DATABASE_URL='postgresql://USER:PASSWORD@HOST/merchantshield?sslmode=require' \
PYTHONPATH=ml/src:services/api \
.venv/bin/alembic -c services/api/alembic.ini upgrade head
```

Then run `alembic current` with the same environment and verify it matches `alembic heads`. Keep the
URL in the backend host's secret manager and URL-encode reserved characters in credentials.

For Supabase, use its direct/session connection for a continuously running FastAPI service. Use its
transaction-mode pooler only for short-lived/serverless API runtimes and follow the provider's
prepared-statement and connection-lifetime guidance. SQLAlchemy still receives only `DATABASE_URL`.

The OpenAI Sites frontend cannot open a raw PostgreSQL TCP connection. It should call the deployed
FastAPI HTTPS origin through `NEXT_PUBLIC_API_URL`; FastAPI alone connects to managed PostgreSQL.
No application-code change is needed when `DATABASE_URL` changes from local Docker to a hosted
provider.

## Troubleshooting

- **Docker is not running:** start Docker Desktop, then rerun `docker compose version` and
  `docker compose ps`.
- **Port 5432 is already in use:** change `POSTGRES_PORT` in `.env` and update the local-host port in
  `DATABASE_URL` to match. The container still listens on port 5432 internally.
- **Database authentication failed:** make `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, and
  the credentials embedded in `DATABASE_URL` agree. Existing named volumes retain their original
  initialized credentials; changing `.env` alone does not rewrite database users.
- **Migration failed:** inspect the first Alembic error, run `alembic current` and `alembic heads`,
  and do not stamp, squash, or delete migrations to bypass it.
- **API container cannot connect:** container-to-container URLs must use `postgres`, not `localhost`.
  `localhost` inside the API container refers to the API container itself.
- **Hosted connection fails:** confirm the provider URL includes the required TLS query parameters,
  the database permits the deployment's network path, and the secret was set on FastAPI—not the
  frontend.
