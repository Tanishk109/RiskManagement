# MerchantShield web

This package delegates to the repository-root vinext application used by OpenAI Sites. It exists as the monorepo frontend boundary while avoiding a duplicated dashboard implementation.

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` to use the FastAPI artifact service locally, or set it to the deployed FastAPI origin for hosting. If the service is unavailable, the dashboard presents an explicit evidence error and never substitutes fabricated metrics.
