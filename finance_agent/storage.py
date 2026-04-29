from __future__ import annotations

import json
from pathlib import Path

import boto3
from botocore.client import Config

from .config import Settings
from .models import RunArtifacts
from .utils import ensure_directory


class StorageManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        ensure_directory(settings.local_storage_path)
        ensure_directory(settings.reports_dir)
        ensure_directory(settings.runs_dir)
        self.s3_client = None
        if settings.s3_bucket:
            self.s3_client = boto3.client(
                "s3",
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                config=Config(signature_version="s3v4"),
            )

    def build_report_path(self, run_id: str) -> Path:
        return self.settings.reports_dir / f"finance-report-{run_id}.pdf"

    def save_run_metadata(self, run_id: str, payload: dict) -> Path:
        output_path = self.settings.runs_dir / f"{run_id}.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        return output_path

    def upload_report(self, report_path: Path) -> str | None:
        if not self.s3_client or not self.settings.s3_bucket:
            return None

        key = f"{self.settings.s3_prefix}/{report_path.name}"
        self.s3_client.upload_file(str(report_path), self.settings.s3_bucket, key)

        if self.settings.s3_public_base_url:
            return f"{self.settings.s3_public_base_url.rstrip('/')}/{key}"

        return self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.settings.s3_bucket, "Key": key},
            ExpiresIn=24 * 60 * 60,
        )

    def finalize_artifacts(
        self,
        run_id: str,
        generated_at: str,
        tickers: list[str],
        report_path: Path,
        metadata_path: Path,
        report_url: str | None,
    ) -> RunArtifacts:
        return RunArtifacts(
            run_id=run_id,
            generated_at=generated_at,
            tickers=tickers,
            report_path=str(report_path.resolve()),
            report_url=report_url,
            metadata_path=str(metadata_path.resolve()),
        )
