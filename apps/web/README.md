# MerchantShield web

This package delegates to the repository-root vinext application used by OpenAI Sites. It exists as the monorepo frontend boundary while avoiding a duplicated dashboard implementation.

Set `NEXT_PUBLIC_API_URL=http://localhost:8000` to use the FastAPI service. If it is absent, the hosted same-origin adapter returns an honest unevaluated state and never fabricates metrics.
