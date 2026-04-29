from __future__ import annotations

import mimetypes
import smtplib
from email.message import EmailMessage
from pathlib import Path

from twilio.rest import Client

from .config import Settings
from .models import RunArtifacts


class Notifier:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def send_notifications(self, artifacts: RunArtifacts, summary_text: str) -> None:
        if self.settings.email_enabled:
            self._send_email(artifacts, summary_text)
        if self.settings.whatsapp_enabled:
            self._send_whatsapp(artifacts, summary_text)

    def _send_email(self, artifacts: RunArtifacts, summary_text: str) -> None:
        if not all(
            [
                self.settings.smtp_host,
                self.settings.smtp_username,
                self.settings.smtp_password,
                self.settings.email_from,
                self.settings.email_to,
            ]
        ):
            raise ValueError("Email delivery is enabled but SMTP settings are incomplete.")

        message = EmailMessage()
        message["Subject"] = f"{self.settings.email_subject_prefix} Daily report {artifacts.generated_at[:10]}"
        message["From"] = self.settings.email_from
        message["To"] = ", ".join(self.settings.email_to)
        message.set_content(
            f"{summary_text}\n\n"
            f"Run ID: {artifacts.run_id}\n"
            f"Local report path: {artifacts.report_path}\n"
            f"Remote report URL: {artifacts.report_url or 'Not uploaded'}\n"
        )

        report_file = Path(artifacts.report_path)
        data = report_file.read_bytes()
        mime_type, _ = mimetypes.guess_type(report_file.name)
        maintype, subtype = (mime_type or "application/pdf").split("/", 1)
        message.add_attachment(data, maintype=maintype, subtype=subtype, filename=report_file.name)

        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port) as server:
            server.starttls()
            server.login(self.settings.smtp_username, self.settings.smtp_password)
            server.send_message(message)

    def _send_whatsapp(self, artifacts: RunArtifacts, summary_text: str) -> None:
        if not all(
            [
                self.settings.twilio_account_sid,
                self.settings.twilio_auth_token,
                self.settings.twilio_whatsapp_from,
                self.settings.twilio_whatsapp_to,
            ]
        ):
            raise ValueError("WhatsApp delivery is enabled but Twilio settings are incomplete.")

        client = Client(self.settings.twilio_account_sid, self.settings.twilio_auth_token)
        body = (
            f"{summary_text}\n"
            f"Run ID: {artifacts.run_id}\n"
            f"Report URL: {artifacts.report_url or 'S3 URL not available; check local storage.'}"
        )
        for recipient in self.settings.twilio_whatsapp_to:
            message_kwargs = {
                "from_": self.settings.twilio_whatsapp_from,
                "to": recipient,
                "body": body,
            }
            if artifacts.report_url:
                message_kwargs["media_url"] = [artifacts.report_url]
            client.messages.create(**message_kwargs)
