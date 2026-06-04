import logging
import requests
import json
from backend.auth.email_config import BREVO_API_KEY, SENDER_EMAIL

logger = logging.getLogger(__name__)

def send_otp_email(recipient_email: str, otp: str) -> bool:
    """
    Sends a 6-digit OTP to the provided recipient email address using Brevo API.
    """
    if BREVO_API_KEY == "PASTE_YOUR_BREVO_API_KEY_HERE" or SENDER_EMAIL == "your_registered_brevo_email@example.com":
        logger.warning("Brevo credentials not configured in email_config.py!")
        return False
        
    try:
        url = "https://api.brevo.com/v3/smtp/email"
        headers = {
            "api-key": BREVO_API_KEY,
            "content-type": "application/json",
            "accept": "application/json"
        }
        
        body = f"""
        Hello,
        
        You have requested to reset your password.
        Your 6-digit One Time Password (OTP) is: {otp}
        
        This OTP is valid for the next 10 minutes.
        If you did not request a password reset, please ignore this email.
        
        Regards,
        Support Team
        """
        
        data = {
            "sender": {
                "name": "Invoice AI Support",
                "email": SENDER_EMAIL
            },
            "to": [
                {
                    "email": recipient_email
                }
            ],
            "subject": "Your Password Reset OTP",
            "textContent": body
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        if response.status_code in [200, 201, 202]:
            logger.info(f"OTP email sent successfully to {recipient_email} via Brevo")
            return True
        else:
            logger.error(f"Brevo API failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send OTP email via Brevo: {e}")
        return False
