from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .models import ConclusionSummary, MacroContext, StockAnalysis


class ReportGenerator:
    def __init__(self) -> None:
        self.styles = getSampleStyleSheet()
        self.styles.add(
            ParagraphStyle(
                name="TitleCenter",
                parent=self.styles["Title"],
                alignment=TA_CENTER,
                textColor=colors.HexColor("#0B1F3A"),
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SectionHeading",
                parent=self.styles["Heading1"],
                textColor=colors.HexColor("#13315C"),
                spaceBefore=12,
            )
        )
        self.styles.add(
            ParagraphStyle(
                name="SubHeading",
                parent=self.styles["Heading2"],
                textColor=colors.HexColor("#24527A"),
                spaceBefore=8,
            )
        )

    def generate(
        self,
        output_path: Path,
        report_date: str,
        tickers: list[str],
        macro_context: MacroContext,
        macro_summary: str,
        analyses: list[StockAnalysis],
        conclusion: ConclusionSummary,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            leftMargin=0.7 * inch,
            rightMargin=0.7 * inch,
            topMargin=0.7 * inch,
            bottomMargin=0.7 * inch,
            title=f"Daily Finance AI Report - {report_date}",
            author="Finance AI Agent",
        )

        story = []
        story.extend(self._build_title_page(report_date, tickers))
        story.extend(self._build_macro_section(macro_context, macro_summary))
        for analysis in analyses:
            story.extend(self._build_stock_section(analysis))
        story.extend(self._build_conclusion_section(conclusion))

        doc.build(story, onFirstPage=self._page_footer, onLaterPages=self._page_footer)
        return output_path

    def _build_title_page(self, report_date: str, tickers: list[str]) -> list:
        return [
            Spacer(1, 1.5 * inch),
            Paragraph("Daily Equity Intelligence Report", self.styles["TitleCenter"]),
            Spacer(1, 0.2 * inch),
            Paragraph(report_date, self.styles["Heading2"]),
            Spacer(1, 0.1 * inch),
            Paragraph("Stocks analyzed: " + ", ".join(tickers), self.styles["BodyText"]),
            Spacer(1, 0.2 * inch),
            Paragraph(
                "This report summarizes the previous 24 hours of company-specific and macroeconomic developments, "
                "then translates them into actionable, research-style commentary.",
                self.styles["BodyText"],
            ),
            PageBreak(),
        ]

    def _build_macro_section(self, macro_context: MacroContext, macro_summary: str) -> list:
        story = [Paragraph("Macro Environment", self.styles["SectionHeading"])]
        story.append(Paragraph(macro_summary, self.styles["BodyText"]))
        story.append(Spacer(1, 0.12 * inch))

        if macro_context.indicators:
            table_data = [["Indicator", "Value", "Date", "Interpretation"]]
            for indicator in macro_context.indicators:
                table_data.append(
                    [indicator.name, indicator.value, indicator.date, indicator.interpretation]
                )
            table = Table(table_data, colWidths=[1.6 * inch, 0.9 * inch, 1.0 * inch, 3.0 * inch])
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#13315C")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#EEF4FB")]),
                    ]
                )
            )
            story.append(table)
            story.append(Spacer(1, 0.16 * inch))

        if macro_context.news:
            story.append(Paragraph("Key Macro Headlines", self.styles["SubHeading"]))
            for item in macro_context.news[:8]:
                story.append(
                    Paragraph(
                        f"<b>{item.title}</b> ({item.source}, {item.published_at})<br/>{item.summary}",
                        self.styles["BodyText"],
                    )
                )
                story.append(Spacer(1, 0.08 * inch))

        story.append(PageBreak())
        return story

    def _build_stock_section(self, analysis: StockAnalysis) -> list:
        story = [Paragraph(f"{analysis.company_name} ({analysis.ticker})", self.styles["SectionHeading"])]
        story.append(Paragraph(analysis.executive_summary, self.styles["BodyText"]))
        story.append(Spacer(1, 0.08 * inch))

        story.append(Paragraph("Sentiment Analysis", self.styles["SubHeading"]))
        story.append(
            Paragraph(
                f"Overall sentiment: <b>{analysis.sentiment.title()}</b> "
                f"(score: {analysis.sentiment_score}/100, confidence: {analysis.confidence.title()})",
                self.styles["BodyText"],
            )
        )

        story.append(Paragraph("Key Events", self.styles["SubHeading"]))
        for event in analysis.key_events:
            story.append(Paragraph(f"- {event}", self.styles["BodyText"]))

        story.append(Paragraph("Impact On Stock Price", self.styles["SubHeading"]))
        story.append(Paragraph(analysis.price_impact, self.styles["BodyText"]))

        story.append(Paragraph("Macro Correlation", self.styles["SubHeading"]))
        story.append(Paragraph(analysis.macro_correlation, self.styles["BodyText"]))

        story.append(Paragraph("Future Outlook", self.styles["SubHeading"]))
        story.append(Paragraph(f"<b>Short term:</b> {analysis.short_term_outlook}", self.styles["BodyText"]))
        story.append(Paragraph(f"<b>Long term:</b> {analysis.long_term_outlook}", self.styles["BodyText"]))

        story.append(Paragraph("Risks", self.styles["SubHeading"]))
        for risk in analysis.risks:
            story.append(Paragraph(f"- {risk}", self.styles["BodyText"]))

        story.append(Paragraph("Opportunities", self.styles["SubHeading"]))
        for opportunity in analysis.opportunities:
            story.append(Paragraph(f"- {opportunity}", self.styles["BodyText"]))

        if analysis.cited_sources:
            sources = "<br/>".join(analysis.cited_sources[:8])
            story.append(Paragraph("Sources", self.styles["SubHeading"]))
            story.append(Paragraph(sources, self.styles["BodyText"]))

        story.append(PageBreak())
        return story

    def _build_conclusion_section(self, conclusion: ConclusionSummary) -> list:
        story = [Paragraph("Final Conclusion", self.styles["SectionHeading"])]
        story.append(Paragraph(conclusion.overview, self.styles["BodyText"]))
        story.append(Paragraph("Top Risks", self.styles["SubHeading"]))
        for item in conclusion.top_risks:
            story.append(Paragraph(f"- {item}", self.styles["BodyText"]))
        story.append(Paragraph("Top Opportunities", self.styles["SubHeading"]))
        for item in conclusion.top_opportunities:
            story.append(Paragraph(f"- {item}", self.styles["BodyText"]))
        story.append(Paragraph("Watch List", self.styles["SubHeading"]))
        for item in conclusion.watch_items:
            story.append(Paragraph(f"- {item}", self.styles["BodyText"]))
        return story

    def _page_footer(self, canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(doc.leftMargin, 20, "Generated by Finance AI Agent")
        canvas.drawRightString(A4[0] - doc.rightMargin, 20, f"Page {doc.page}")
        canvas.restoreState()
