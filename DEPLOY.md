# Production deployment (Vercel + API)

## Architecture

- **Frontend (`upload-web`)** — deploy to [Vercel](https://vercel.com). Next.js is a native fit for Vercel.
- **Backend (`upload-api`)** — keep on a container or VM host (Dockerfile provided). FastAPI + pandas + scikit-learn + LangGraph need a **stateful disk or object store** for `uploads/` and are a poor match for short Vercel serverless timeouts.

## 1. API (Docker)

From `upload-api/`:

```bash
docker build -t ads-api .
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=sk-... \
  -e OPENAI_MODEL=gpt-4o-mini \
  -e CORS_ORIGINS=https://your-app.vercel.app \
  ads-api
```

Set `CORS_ORIGINS` to your exact Vercel URL(s). The default `CORS_ORIGIN_REGEX` already allows any `https://*.vercel.app` preview URL.

Health check: `GET /health` includes `llm_configured`.

## 2. Vercel (frontend)

1. Import the Git repo in Vercel.
2. Set **Root Directory** to `upload-web`.
3. Environment variable: `NEXT_PUBLIC_API_URL=https://your-api-host.example.com` (no trailing slash).
4. Deploy.

## 3. Secrets

- **Never** put `OPENAI_API_KEY` in the Next.js bundle. It belongs only on the Python API.
- Optional: add observability and rate limits on the API before exposing `/analyze-llm` publicly.
