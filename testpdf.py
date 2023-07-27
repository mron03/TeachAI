import os
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO

def generate_pdf(text):
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    # Register the "DejaVuSans" font, which supports Cyrillic characters
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

    # Create a custom style for your text using the "DejaVuSans" font
    custom_style = ParagraphStyle(
        'CustomStyle',
        parent=styles['Normal'],
        fontName='DejaVuSans',
        fontSize=12,
        textColor=colors.black,
        spaceAfter=12,
    )

    story = []

    # Add your generated text to the story
    for paragraph in text.split('\n'):
        p = Paragraph(paragraph, custom_style)
        story.append(p)

    doc.build(story)
    buffer.seek(0)

    return buffer

def read_pdf_content(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        pdf_text = []
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            pdf_text.append(page.extract_text())

    return pdf_text

# Your text content
text = '''Instruction 1 : Введение в тему
Speech 1 : Сегодня мы будем изучать тему 'Что такое Lorem Ipsum?'
Instruction 2 : Объяснение Lorem Ipsum
Speech 2 : Lorem Ipsum - это просто фиктивный текст печати и наборной промышленности. Lorem Ipsum является стандартным фиктивным текстом отрасли с момента 1500-х годов, когда неизвестный печатник взял галью типа и перемешал его, чтобы создать образец типа. Он выжил не только пять веков, но и пережил переход в электронный набор, оставаясь в сущности неизменным. Он стал популярным в 1960-х годах с выпуском листов Letraset, содержащих отрывки Lorem Ipsum, и недавно с программным обеспечением по визуальному оформлению, таким как Aldus PageMaker, включая версии Lorem Ipsum.
Instruction 3 : Примеры и задачи
Speech 3 : Давайте рассмотрим примеры и задачи, чтобы лучше понять тему. Вот пример задачи:
Instruction 4 : Резюме
Speech 4 : В заключение, Lorem Ipsum - это фиктивный текст, используемый в печати и наборной промышленности. Он был популяризирован в 1960-х годах и остается популярным по сей день.'''

# Generate the PDF
pdf = generate_pdf(text)

# Save the PDF to a file
pdf_path = 'output.pdf'
with open(pdf_path, 'wb') as f:
    f.write(pdf.read())

# Read and print the content of the PDF
pdf_content = read_pdf_content(pdf_path)
for page_num, page_text in enumerate(pdf_content, 1):
    print(f"Page {page_num}:\n{page_text}\n")
