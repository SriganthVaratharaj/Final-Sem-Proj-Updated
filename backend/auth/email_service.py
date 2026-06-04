import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from backend.auth.email_config import SENDER_EMAIL, APP_PASSWORD

logger = logging.getLogger(__name__)

def send_otp_email(recipient_email: str, otp: str) -> bool:
    """
    Sends a 6-digit OTP to the provided recipient email address.
    """
    if SENDER_EMAIL == "your_email@gmail.com" or APP_PASSWORD == "your_app_password":
        logger.warning("Email credentials not configured in email_config.py!")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = recipient_email
        msg['Subject'] = "Your Password Reset OTP"

        body = f"""
        Hello,
        
        You have requested to reset your password.
        Your 6-digit One Time Password (OTP) is: {otp}
        
        This OTP is valid for the next 10 minutes.
        If you did not request a password reset, please ignore this email.
        
        Regards,
        Support Team
        """
        
        msg.attach(MIMEText(body, 'plain'))

        # Connect to Gmail SMTP server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        
        text = msg.as_string()
        server.sendmail(SENDER_EMAIL, recipient_email, text)
        server.quit()
        
        logger.info(f"OTP email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send OTP email: {e}")
        return False
