import io
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        self.drawString(54, letter[1] - 36, "Automated NMR Structure Elucidation Platform")
        self.drawRightString(letter[0] - 54, letter[1] - 36, "CONFIDENTIAL / ANALYTICAL DATA")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        self.drawRightString(letter[0] - 54, 36, f"Page {self._pageNumber} of {count}")
        self.line(54, 46, letter[0] - 54, 46)
        self.restoreState()

def build_pdf_report(sample_id: str, smiles: str, solvent: str, freq: str, df_1h: pd.DataFrame, df_13c: pd.DataFrame, img_buf: io.BytesIO) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()

    t_style = ParagraphStyle('DocTitle', parent=styles['Heading1'], fontName='Helvetica-Bold', fontSize=16, textColor=colors.HexColor("#0B3C5D"))
    s_style = ParagraphStyle('SecTitle', parent=styles['Heading2'], fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor("#1D2731"), spaceBefore=8, spaceAfter=4)
    c_style = ParagraphStyle('Cell', parent=styles['Normal'], fontName='Helvetica', fontSize=8, alignment=1)
    h_style = ParagraphStyle('Hdr', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, textColor=colors.white, alignment=1)

    story = [Paragraph("Structure Elucidation & NMR Assignment Report", t_style), Spacer(1, 4)]

    meta = [
        [Paragraph(f"<b>Sample ID:</b> {sample_id}", styles['Normal']), Paragraph(f"<b>Solvent:</b> {solvent}", styles['Normal'])],
        [Paragraph(f"<b>SMILES:</b> {smiles}", styles['Normal']), Paragraph(f"<b>Spectrometer:</b> {freq}", styles['Normal'])]
    ]
    t_meta = Table(meta, colWidths=[3.5 * inch, 3.5 * inch])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F5F7FA")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.extend([t_meta, Spacer(1, 6)])

    # Chemical Structure
    story.append(Paragraph("1. Numbered Chemical Structure", s_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B3C5D"), spaceAfter=6))
    img = Image(img_buf, width=3.2 * inch, height=2.4 * inch)
    t_img = Table([[img]], colWidths=[7.0 * inch])
    t_img.setStyle(TableStyle([('ALIGN', (0,0), (-1,-1), 'CENTER')]))
    story.extend([t_img, Spacer(1, 6)])

    # 1H Table
    story.append(Paragraph("2. ¹H NMR Assignment Table", s_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0B3C5D"), spaceAfter=4))
    h_hdr = ["Atom", "Pred δ", "Exp δ", "Range", "Mult.", "Int.", "Status"]
    h_rows = [[Paragraph(h, h_style) for h in h_hdr]]
    for _, r in df_1h.iterrows():
        h_rows.append([
            Paragraph(str(r["Atom Label"]), c_style),
            Paragraph(f"{r['Pred δ (ppm)']:.2f}", c_style),
            Paragraph(f"{r['Exp δ (ppm)']:.2f}" if pd.notna(r['Exp δ (ppm)']) else "-", c_style),
            Paragraph(str(r["Range (ppm)"]), c_style),
            Paragraph(str(r["Mult."]), c_style),
            Paragraph(str(r["Integral"]), c_style),
            Paragraph(str(r["Status"]), c_style)
        ])
    t_h = Table(h_rows, colWidths=[1.0*inch, 0.9*inch, 0.9*inch, 1.4*inch, 0.8*inch, 0.8*inch, 1.2*inch])
    t_h.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#0B3C5D")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.extend([t_h, Spacer(1, 8)])

    # 13C Table
    story.append(KeepTogether([
        Paragraph("3. ¹³C NMR Assignment Table", s_style),
        HRFlowable(width="100%", thickness=1, color=colors.HexColor("#328CC1"), spaceAfter=4)
    ]))
    c_hdr = ["Atom", "Type", "Pred δ", "Exp δ", "Status"]
    c_rows = [[Paragraph(h, h_style) for h in c_hdr]]
    for _, r in df_13c.iterrows():
        c_rows.append([
            Paragraph(str(r["Atom Label"]), c_style),
            Paragraph(str(r["Type"]), c_style),
            Paragraph(f"{r['Pred δ (ppm)']:.1f}", c_style),
            Paragraph(f"{r['Exp δ (ppm)']:.1f}" if pd.notna(r['Exp δ (ppm)']) else "-", c_style),
            Paragraph(str(r["Status"]), c_style)
        ])
    t_c = Table(c_rows, colWidths=[1.4*inch, 1.4*inch, 1.4*inch, 1.4*inch, 1.4*inch])
    t_c.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#328CC1")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#D1D5DB")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F9FAFB")]),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    story.append(KeepTogether([t_c]))

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()
