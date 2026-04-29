# Finance AI Agent

Production-ready Python service that runs every day at 12:00 AM IST, collects the last 24 hours of company and macroeconomic signals, uses the OpenAI API to generate equity-style analysis, renders a PDF report, and delivers it by email and WhatsApp.

## Features

- Configurable stock coverage using `TICKERS`
- News, filings, market snapshot, and macro data collection
- LLM-driven research note generation with heuristic fallbacks
- Professional PDF output via ReportLab
- Email delivery with PDF attachment
- WhatsApp delivery through Twilio, with optional S3-hosted PDF links
- Daily scheduling with APScheduler
- Persistent logs and JSON run metadata
- Docker-ready deployment for EC2, Railway, or Render

## Project Structure

```text
finance_agent/
  analyzer.py
  config.py
  data_fetcher.py
  logging_config.py
  main.py
  models.py
  notifier.py
  report_generator.py
  scheduler.py
  storage.py
  utils.py
main.py
requirements.txt
Dockerfile
.env.example
README.md
```

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill in your keys and recipients.

```bash
cp .env.example .env
```

4. Run once for a manual test.

```bash
python main.py run-once
```

5. Start the long-running scheduler.

```bash
python main.py schedule
```

## Environment Variables

Required for core analysis:

- `TICKERS`
- `OPENAI_API_KEY`

Recommended for richer data:

- `NEWSAPI_API_KEY`
- `FINNHUB_API_KEY`
- `ALPHA_VANTAGE_API_KEY`

Optional delivery:

- `EMAIL_ENABLED=true` plus SMTP credentials
- `WHATSAPP_ENABLED=true` plus Twilio credentials
- `S3_BUCKET` if you want cloud-hosted report links for WhatsApp delivery

## How It Works

1. `data_fetcher.py` loads market data from Yahoo Finance, company news from Yahoo Finance and NewsAPI, filings from Finnhub, and macro indicators from Alpha Vantage.
2. `analyzer.py` sends structured prompts to the OpenAI API and produces stock-by-stock analysis plus a portfolio conclusion.
3. `report_generator.py` builds a PDF with a title page, macro section, stock sections, and final conclusion.
4. `storage.py` saves the PDF and JSON run metadata locally, then optionally uploads the report to S3.
5. `notifier.py` sends the report by email and optionally pushes the summary and report link through Twilio WhatsApp.
6. `scheduler.py` runs the pipeline every day at `RUN_HOUR:RUN_MINUTE` in `TIMEZONE`.

## Running At 12:00 AM IST

Default settings already target midnight India time:

```env
TIMEZONE=Asia/Kolkata
RUN_HOUR=0
RUN_MINUTE=0
```

This is used by the built-in APScheduler service when you run:

```bash
python main.py schedule
```

## Cron Example

If you prefer server cron instead of the built-in scheduler, keep the app in `run-once` mode and schedule it externally.

Linux cron example:

```cron
TZ=Asia/Kolkata
0 0 * * * cd /opt/finance-ai-agent && /opt/finance-ai-agent/.venv/bin/python main.py run-once >> /opt/finance-ai-agent/artifacts/logs/cron.log 2>&1
```

## Docker

Build and run:

```bash
docker build -t finance-ai-agent .
docker run --env-file .env -v $(pwd)/artifacts:/app/artifacts finance-ai-agent
```

The container default command is:

```bash
python main.py schedule
```

If you want one-shot execution inside a scheduler platform:

```bash
docker run --env-file .env -v $(pwd)/artifacts:/app/artifacts finance-ai-agent python main.py run-once
```

## Deployment Guide

### AWS EC2

1. Launch a small Ubuntu instance.
2. Install Docker or Python 3.12.
3. Copy the project and your `.env`.
4. Run the container with restart policy:

```bash
docker run -d \
  --name finance-ai-agent \
  --restart unless-stopped \
  --env-file /opt/finance-ai-agent/.env \
  -v /opt/finance-ai-agent/artifacts:/app/artifacts \
  finance-ai-agent
```

### Railway

1. Create a new service from this repository or uploaded project.
2. Add all `.env` variables in Railway's Variables tab.
3. Use the default Dockerfile.
4. Keep the start command as `python main.py schedule`.

### Render

1. Create a new Background Worker service.
2. Point it to this codebase or upload it.
3. Set environment variables from `.env.example`.
4. Use the Dockerfile or a Python environment with start command `python main.py schedule`.

## Delivery Notes

- Gmail generally requires an app password if 2FA is enabled.
- Twilio WhatsApp media delivery works best when the PDF has a public or presigned URL, so configure `S3_BUCKET` for production WhatsApp usage.
- If S3 is not configured, WhatsApp messages still send a summary but will not include an attachment.

## Logs And Persistence

- Log file: `artifacts/logs/finance_agent.log`
- PDF reports: `artifacts/reports/`
- Run metadata JSON: `artifacts/runs/`

Each run metadata file stores fetched context, generated analysis, report path, report URL, and any per-ticker failures.

## Production Hardening Suggestions

- Store secrets in AWS Secrets Manager, Railway Variables, or Render Environment Groups instead of local `.env` in production.
- Put the PDF bucket behind least-privilege IAM credentials.
- Add application monitoring such as CloudWatch, Datadog, or Sentry.
- Add a dead-letter or alerting mechanism for repeated delivery failures.
