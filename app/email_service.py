import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class EmailService:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")
        self.email_to = os.getenv("EMAIL_TO")

        if not all([self.smtp_user, self.smtp_pass, self.email_to]):
            logger.warning(
                "Email configuration incomplete. Email notifications will be disabled."
            )

    def send_rag_notification(
        self, question: str, answer: str, sources: list
    ) -> Dict[str, Any]:
        """
        Send email notification with RAG question and answer

        Returns:
            Dict with status and message
        """
        if not all([self.smtp_user, self.smtp_pass, self.email_to]):
            return {"status": "failed", "message": "Email configuration incomplete"}

        try:
            # Create message
            msg = MIMEMultipart()
            msg["From"] = self.smtp_user
            msg["To"] = self.email_to
            msg["Subject"] = "RAG message"

            # Create email body
            body = f"Question: {question}\n\nAnswer: {answer}"

            if sources:
                body += f"\n\nSources: {', '.join(sources)}"

            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)

            logger.info(f"Email sent successfully to {self.email_to}")
            return {"status": "success", "message": f"Email sent to {self.email_to}"}

        except Exception as e:
            logger.error(f"Failed to send email: {str(e)}")
            return {"status": "failed", "message": f"Email failed: {str(e)}"}


# Global instance
email_service = EmailService()
