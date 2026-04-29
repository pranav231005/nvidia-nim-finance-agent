import textwrap

line = "------------------------------------------------------------------------------------------------------------------------------------------------------------"
clean_line = textwrap.fill(line, width=90, break_long_words=True)
print(repr(clean_line))

from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("helvetica", size=11)
try:
    pdf.multi_cell(0, 8, clean_line)
    print("PDF SUCCESS")
except Exception as e:
    print(f"PDF FAILED: {e}")
