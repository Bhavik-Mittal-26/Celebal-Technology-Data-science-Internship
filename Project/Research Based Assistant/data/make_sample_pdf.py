"""
Generates a sample HR Policy PDF used to test/demo the RAG assistant.
Run once: python make_sample_pdf.py
"""
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

doc = SimpleDocTemplate("sample_hr_policy.pdf", pagesize=letter)
styles = getSampleStyleSheet()
heading = ParagraphStyle('Heading', parent=styles['Heading1'], spaceAfter=10)
body = styles['Normal']

story = []
story.append(Paragraph("Acme Corp — Employee Handbook", styles['Title']))
story.append(Spacer(1, 20))

sections = [
    ("1. Leave Policy", """
    Employees are entitled to 18 days of paid annual leave per calendar year, accrued
    monthly at 1.5 days per month. Unused leave up to 10 days may be carried forward
    to the next year; any excess is forfeited. Leave requests must be submitted at
    least 3 working days in advance via the HR portal, except in emergencies.
    """),
    ("2. Maternity Leave Policy", """
    Female employees are entitled to 26 weeks of paid maternity leave for the first
    two children, and 12 weeks for the third child onward, in accordance with
    applicable labor law. Maternity leave may begin up to 8 weeks before the expected
    delivery date. Employees must notify HR at least 60 days before the leave starts,
    along with a medical certificate confirming the expected delivery date. On
    return, employees are guaranteed the same or an equivalent role and salary.
    """),
    ("3. Paternity Leave Policy", """
    Male employees are entitled to 2 weeks of paid paternity leave, to be taken
    within 3 months of the child's birth or adoption. Requests must be submitted to
    HR at least 2 weeks in advance where possible.
    """),
    ("4. Sick Leave Policy", """
    Employees receive 12 paid sick leave days per year. A medical certificate is
    required for sick leave exceeding 2 consecutive days. Sick leave does not carry
    forward to the next calendar year.
    """),
    ("5. Work From Home Policy", """
    Employees may work from home up to 2 days per week, subject to manager approval.
    Roles requiring on-site presence (e.g., lab, manufacturing) are excluded from
    this policy. Employees must remain reachable during standard working hours
    (9:30 AM to 6:30 PM) on WFH days.
    """),
    ("6. Code of Conduct", """
    All employees are expected to act with integrity, respect confidentiality of
    company and client data, and avoid conflicts of interest. Harassment or
    discrimination of any kind is strictly prohibited and will result in
    disciplinary action, up to and including termination.
    """),
    ("7. Expense Reimbursement Policy", """
    Business-related expenses (travel, client meals, approved software) are
    reimbursed within 15 working days of submission through the finance portal.
    Claims must be submitted with original receipts within 30 days of the expense
    being incurred. Expenses without receipts above INR 500 will not be reimbursed.
    """),
    ("8. Termination and Notice Period", """
    Employees are required to serve a notice period of 60 days upon resignation.
    The company may, at its discretion, provide pay in lieu of notice. During the
    probation period (first 6 months of employment), the notice period is reduced
    to 15 days for both employee and employer.
    """),
]

for title, text in sections:
    story.append(Paragraph(title, heading))
    story.append(Paragraph(text.strip().replace("\n", " "), body))
    story.append(Spacer(1, 14))

doc.build(story)
print("Created sample_hr_policy.pdf")
