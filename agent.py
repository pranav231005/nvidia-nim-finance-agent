import os
import datetime
import yfinance as yf
from fpdf import FPDF
import google.generativeai as genai
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# Configuration
TICKERS = os.getenv("TICKERS", "AAPL,MSFT,TSLA").split(",")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Email configuration (Optional, for sending the PDF)
EMAIL_SENDER = os.getenv("EMAIL_SENDER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD") 
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=GEMINI_API_KEY)

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

def analyze_with_gemini(all_data):
    print("Analyzing data with Gemini...")
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    
    date_today = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")
    
    prompt = f"""
    You are an expert financial analyst AI agent.
    Analyze the following stocks using the latest available information from the past 24 hours.
    
    Current Date and Time: {date_today}
    
    Data:
    {all_data}
    
    INSTRUCTIONS:
    1. DATA COLLECTION: Consider the provided news and general macroeconomic factors (Fed rates, inflation, GDP).
    2. ANALYSIS: For each stock summarize key events, perform sentiment analysis (Positive/Negative/Neutral), identify short/long-term drivers, and evaluate risk factors.
    3. MARKET IMPACT: Predict likely short-term stock movement (Bullish/Bearish/Neutral) and explain WHY. Discuss long-term outlook.
    4. OUTPUT FORMAT:
       For each stock:
       - Company Name
       - Key Updates (last 24 hrs)
       - Sentiment Analysis
       - Expected Short-Term Impact
       - Long-Term Outlook
       - Risk Factors
    5. FINAL SUMMARY: Overall market sentiment, key macroeconomic drivers, and portfolio-level conclusion.
    6. STYLE: Professional financial report tone, precise, data-driven. Use markdown formatting.
    """
    
    response = model.generate_content(prompt)
    return response.text

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Daily Financial Analysis Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(markdown_text, filename="Financial_Report.pdf"):
    print("Generating PDF...")
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # Very basic markdown to text conversion for FPDF (which doesn't natively support markdown)
    # In a production app, you might want to use a more robust markdown-to-pdf library like xhtml2pdf or weasyprint
    for line in markdown_text.split('\n'):
        # Handle basic markdown bolding
        clean_line = line.replace('**', '').replace('##', '').replace('*', '')
        # Handle encoding issues
        clean_line = clean_line.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, clean_line)
        
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

def main():
    all_stock_data = ""
    for ticker in TICKERS:
        news = fetch_stock_data(ticker.strip())
        all_stock_data += f"\n--- {ticker.strip()} ---\n{news}\n"
        
    analysis_result = analyze_with_gemini(all_stock_data)
    
    # Save raw markdown as artifact
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(analysis_result)
        
    pdf_filename = f"Financial_Report_{datetime.datetime.now().strftime('%Y%m%d')}.pdf"
    create_pdf(analysis_result, pdf_filename)
    
    send_email(pdf_filename)

if __name__ == "__main__":
    main()
