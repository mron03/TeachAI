import pdfkit
import random

def generate_text():
  """Generates a random text of 1000 characters."""
  text = ""
  for i in range(1000):
    char = chr(random.randint(ord("a"), ord("z")))
    text += char
  return text

def download_pdf(text):
  """Downloads a PDF file containing the given text."""
  pdfkit.from_string(text, "output.pdf")

if __name__ == "__main__":
  text = generate_text()
  download_pdf(text)
