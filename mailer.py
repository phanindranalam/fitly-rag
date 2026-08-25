"""Optional "email me these matches" delivery over SMTP.

Standard library only, and entirely inert unless SMTP secrets are set, so a
deployment without mail configured shows no broken button. Nothing is stored:
the address is used for one send and never written anywhere.
"""

from __future__ import annotations

import html
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import streamlit as st

REQUIRED = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM")


def available() -> bool:
    try:
        return all(bool(st.secrets.get(k)) for k in REQUIRED)
    except Exception:
        return False


def build_html(question: str, answer_text: str, hits: list, refused: bool) -> str:
    rows = []
    for i, h in enumerate(hits, start=1):
        loc = f" &middot; {html.escape(h.location)}" if h.location else ""
        pay = f" &middot; {html.escape(h.compensation)}" if h.compensation else ""
        link = (f'<a href="{html.escape(h.url)}">{html.escape(h.title or "role")}</a>'
                if h.url else html.escape(h.title or "role"))
        rows.append(
            f'<li style="margin-bottom:10px"><strong>[{i}] {html.escape(h.company or "")}</strong> '
            f'&mdash; {link}{loc}{pay}</li>')

    body = html.escape(answer_text).replace("\n", "<br>")
    heading = "No grounded answer" if refused else "What the postings say"
    return f"""<html><body style="font-family:-apple-system,Segoe UI,sans-serif;
      color:#1a1a1a;max-width:640px">
      <h2 style="margin-bottom:4px">Fitly RAG</h2>
      <p style="color:#666;margin-top:0">{html.escape(question)}</p>
      <h3>{heading}</h3>
      <p>{body}</p>
      <h3>Sources</h3>
      <ol>{''.join(rows) or '<li>None</li>'}</ol>
      <p style="color:#888;font-size:12px">Every claim above is drawn from the
      numbered postings. Nothing was added from outside them.</p>
    </body></html>"""


def send(to_email: str, subject: str, html_body: str) -> tuple[bool, str]:
    """Returns (ok, message). Never raises."""
    if not available():
        return False, "Email isn't configured on this deployment."
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = st.secrets["SMTP_FROM"]
        msg["To"] = to_email
        msg.attach(MIMEText(html_body, "html"))
        with smtplib.SMTP(st.secrets["SMTP_HOST"], int(st.secrets["SMTP_PORT"]), timeout=15) as s:
            s.starttls()
            s.login(st.secrets["SMTP_USER"], st.secrets["SMTP_PASSWORD"])
            s.send_message(msg)
        return True, f"Sent to {to_email}."
    except smtplib.SMTPAuthenticationError:
        return False, "The mail server rejected the app's credentials."
    except (smtplib.SMTPException, OSError, ValueError) as e:
        return False, f"Couldn't send right now ({type(e).__name__})."
