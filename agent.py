import os
import datetime
import yfinance as yf
from fpdf import FPDF
from google import genai
from openai import OpenAI
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import requests
from twilio.rest import Client

# Configuration
tickers_env = os.getenv("TICKERS", "").strip()
if not tickers_env:
    tickers_env = "RELIANCE.NS,TCS.NS,HDFCBANK.NS"
TICKERS = tickers_env.split(",")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
NIM_API_KEY = os.getenv("NIM_API_KEY")
NIM_BASE_URL = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
NIM_MODEL = os.getenv("NIM_MODEL", "deepseek-ai/deepseek-v4-pro")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  # "gemini" or "nim"

# Email configuration (Optional, for sending the PDF)
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") 
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

# WhatsApp configuration (Twilio)
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_WHATSAPP_SENDER = os.getenv("TWILIO_WHATSAPP_SENDER")
WHATSAPP_RECEIVER = os.getenv("WHATSAPP_RECEIVER")
if not GEMINI_API_KEY and LLM_PROVIDER == "gemini":
    raise ValueError("GEMINI_API_KEY environment variable is not set.")
if not NIM_API_KEY and LLM_PROVIDER == "nim":
    raise ValueError("NIM_API_KEY environment variable is not set.")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
nim_client = OpenAI(api_key=NIM_API_KEY, base_url=NIM_BASE_URL) if NIM_API_KEY else None

def fetch_stock_data(ticker):
    print(f"Fetching data for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return f"Could not fetch news due to an error: {e}"
    
    # Format news into a readable string
    news_str = ""
    if news:
        for item in news[:5]: # Get top 5 news articles
            title = item.get('title', 'No Title')
            publisher = item.get('publisher', 'Unknown')
            news_str += f"- {title} ({publisher})\n"
    else:
        news_str = "No recent news found."
        
    return news_str

def analyze_with_nim(all_data):
    print("Analyzing data with NVIDIA NIM (DeepSeek V4 Pro)...")

    date_today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    prompt = f"""
    You are an expert financial analyst AI agent.
    Provide a DEEP narrative analysis for each of the following stocks using the latest available information from the past 24 hours.

    Current Date and Time: {date_today}

    Data:
    {all_data}

    INSTRUCTIONS:
    1. DEEP ANALYSIS: DO NOT generate tables. Provide a detailed narrative for EACH stock. Discuss any news, earnings results, contracts, or events from the past 24 hours.
    2. EVALUATION: Discuss the short-term impact of these events and the long-term outlook. Identify any critical risk factors.
    3. FORMATTING: Use Markdown.
       Format EACH stock EXACTLY like this:
       ## Company Name (Ticker)
       - **24-Hour Update:** (Detailed explanation of news/results)
       - **Short-Term Impact:** (Analysis of price action and sentiment)
       - **Long-Term Outlook:** (Fundamental view)
       - **Key Risks:** (What to watch out for)

    4. NO TABLES: Under no circumstances should you generate a markdown table.
    5. FINAL SUMMARY: End with a "## Market Overview" section summarizing the broader market sentiment and key macroeconomic drivers.
    """

    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = nim_client.chat.completions.create(
                model=NIM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API busy (attempt {attempt + 1}/{max_retries}). Waiting 30 seconds... Error: {e}")
                import time
                time.sleep(30)
            else:
                print("Failed after maximum retries.")
                raise e


def generate_analysis(all_data):
    if LLM_PROVIDER == "nim":
        return analyze_with_nim(all_data)
    else:
        return analyze_with_gemini(all_data)


def analyze_with_gemini(all_data):
    print("Analyzing data with Gemini...")

    date_today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    prompt = f"""
    You are an expert financial analyst AI agent.
    Provide a DEEP narrative analysis for each of the following stocks using the latest available information from the past 24 hours.
    
    Current Date and Time: {date_today}
    
    Data:
    {all_data}
    
    INSTRUCTIONS:
    1. DEEP ANALYSIS: DO NOT generate tables. Provide a detailed narrative for EACH stock. Discuss any news, earnings results, contracts, or events from the past 24 hours.
    2. EVALUATION: Discuss the short-term impact of these events and the long-term outlook. Identify any critical risk factors.
    3. FORMATTING: Use Markdown. 
       Format EACH stock EXACTLY like this:
       ## Company Name (Ticker)
       - **24-Hour Update:** (Detailed explanation of news/results)
       - **Short-Term Impact:** (Analysis of price action and sentiment)
       - **Long-Term Outlook:** (Fundamental view)
       - **Key Risks:** (What to watch out for)
       
    4. NO TABLES: Under no circumstances should you generate a markdown table.
    5. FINAL SUMMARY: End with a "## Market Overview" section summarizing the broader market sentiment and key macroeconomic drivers.
    """
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3.1-flash-lite-preview',
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"API busy (attempt {attempt + 1}/{max_retries}). Waiting 30 seconds... Error: {e}")
                import time
                time.sleep(30)
            else:
                print("Failed after maximum retries.")
                raise e

class PDFReport(FPDF):
    # Minimalist color palette
    TEXT_COLOR = (30, 30, 30)
    LIGHT_TEXT = (120, 120, 120)

    def header(self):
        # Clean white header, left aligned
        self.set_y(15)
        self.set_font('helvetica', 'B', 18)
        self.set_text_color(*self.TEXT_COLOR)
        self.cell(0, 8, 'Daily Financial Analysis Report', new_x="LMARGIN", new_y="NEXT", align='L')
        
        # Subtitle date in light gray
        self.set_font('helvetica', '', 10)
        self.set_text_color(*self.LIGHT_TEXT)
        self.cell(0, 6, f'Generated on {datetime.datetime.now().strftime("%A, %B %d, %Y")}', new_x="LMARGIN", new_y="NEXT", align='L')
        self.ln(6)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', '', 9)
        self.set_text_color(*self.LIGHT_TEXT)
        self.cell(0, 10, str(self.page_no()), align='C')

    def section_header(self, title):
        """Print a styled section header (H1/H2)."""
        self.ln(6)
        self.set_font('helvetica', 'B', 14)
        self.set_text_color(*self.TEXT_COLOR)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", align='L')
        self.ln(2)

    def bullet_line(self, text):
        """Print a standard bullet point."""
        import textwrap
        self.set_font('helvetica', '', 10)
        self.set_text_color(*self.TEXT_COLOR)
        
        # Bullet character
        bullet = chr(149)
        
        wrapped = textwrap.wrap(text, width=90, break_long_words=True)
        for i, w_line in enumerate(wrapped):
            self.set_x(self.l_margin + 5)
            if i == 0:
                self.cell(5, 5.5, bullet, align='L')
                self.cell(0, 5.5, w_line, new_x="LMARGIN", new_y="NEXT")
            else:
                self.set_x(self.l_margin + 10)
                self.cell(0, 5.5, w_line, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)

    def body_line(self, text):
        """Print a regular body line."""
        import textwrap
        self.set_font('helvetica', '', 10)
        self.set_text_color(*self.TEXT_COLOR)
        wrapped = textwrap.wrap(text, width=95, break_long_words=True)
        for w_line in wrapped:
            self.cell(0, 5.5, w_line, new_x="LMARGIN", new_y="NEXT")
        self.ln(1.5)


def create_pdf(markdown_text, filename="Financial_Report.pdf"):
    print("Generating PDF...")
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.set_margins(left=15, top=35, right=15)
    pdf.add_page()

    for line in markdown_text.split('\n'):
        raw = line.strip()
        if not raw:
            pdf.ln(3)
            continue

        # Encode safely
        raw = raw.encode('latin-1', 'replace').decode('latin-1')

        # H1/H2/H3/H4 section headers: start with #
        if raw.startswith('#'):
            title = raw.lstrip('#').strip()
            pdf.section_header(title)

        # Bullet points: - item or * item
        elif raw.startswith('- ') or raw.startswith('* '):
            text = raw[2:].replace('**', '').replace('*', '').strip()
            pdf.bullet_line(text)

        # Bold-only line (markdown heading substitute like **Company Name**)
        elif raw.startswith('**') and raw.endswith('**'):
            title = raw.replace('**', '').strip()
            pdf.set_font('helvetica', 'B', 11)
            pdf.set_text_color(15, 23, 42)
            pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)

        # Separator lines (----)
        elif raw.startswith('---'):
            pdf.set_draw_color(220, 220, 230)
            pdf.set_line_width(0.3)
            pdf.line(pdf.l_margin, pdf.get_y(), 195, pdf.get_y())
            pdf.ln(4)

        # Regular body text
        else:
            clean = raw.replace('**', '').replace('*', '').strip()
            if clean:
                pdf.body_line(clean)

    pdf.output(filename)
    print(f"PDF saved as {filename}")
    return filename


def send_email(filename):
    if not all([EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER]):
        print("Email credentials not fully provided. Skipping email sending.")
        return
        
    print("Sending email...")
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg['Subject'] = f"Daily Financial Report - {datetime.datetime.now().strftime('%Y-%m-%d')}"
    
    body = "Please find attached your daily financial analysis report."
    msg.attach(MIMEText(body, 'plain'))
    
    with open(filename, "rb") as f:
        attach = MIMEApplication(f.read(), _subtype="pdf")
        attach.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(attach)
        
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

def send_whatsapp(filename):
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_SENDER, WHATSAPP_RECEIVER]):
        print("Twilio credentials missing. Skipping WhatsApp.")
        return
        
    print("Uploading PDF to secure temporary storage for WhatsApp...")
    try:
        url = "https://tmpfiles.org/api/v1/upload"
        with open(filename, 'rb') as f:
            response = requests.post(url, files={'file': f})
            
        if response.status_code == 200:
            data = response.json()
            original_url = data['data']['url']
            
            # tmpfiles.org requires /dl/ inserted in the path for direct media downloads
            media_url = original_url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
            # ensure it uses https for Twilio
            media_url = media_url.replace("http://", "https://")
            
            print(f"PDF successfully uploaded to: {media_url}")
            
            print("Sending WhatsApp message via Twilio...")
            client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
            message = client.messages.create(
                from_=TWILIO_WHATSAPP_SENDER,
                body="Here is your Daily Financial Analysis Report!",
                media_url=[media_url],
                to=WHATSAPP_RECEIVER
            )
            print(f"WhatsApp message sent successfully! SID: {message.sid}")
        else:
            print(f"Failed to upload PDF for WhatsApp. Status Code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send WhatsApp message: {e}")
def main():
    all_stock_data = ""
    for ticker in TICKERS:
        news = fetch_stock_data(ticker.strip())
        all_stock_data += f"\n--- {ticker.strip()} ---\n{news}\n"
        
    analysis_result = generate_analysis(all_stock_data)
    
    # Save raw markdown as artifact
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(analysis_result)
        
    pdf_filename = f"Financial_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
    create_pdf(analysis_result, pdf_filename)
    
    send_email(pdf_filename)
    send_whatsapp(pdf_filename)

if __name__ == "__main__":
    main()
