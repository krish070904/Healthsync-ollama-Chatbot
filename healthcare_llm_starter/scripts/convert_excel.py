# scripts/convert_guidelines.py
import pathlib, pandas as pd, sys
from docx import Document  # pip install python-docx
from pathlib import Path

DATA_DIR = Path("data/guidelines")

def txt_from_xlsx(path: Path):
    xls = pd.ExcelFile(path)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet, dtype=str)
        txt = df.fillna('').to_csv(sep='\t', index=False)
        out = DATA_DIR / f"{path.stem}__{sheet}.txt"
        out.write_text(f"Source: {path.name}\nSheet: {sheet}\n\n{txt}", encoding='utf-8')
        print("wrote", out)

def txt_from_csv(path: Path):
    df = pd.read_csv(path, dtype=str)
    out = DATA_DIR / f"{path.stem}.txt"
    out.write_text(df.fillna('').to_csv(sep='\t', index=False), encoding='utf-8')
    print("wrote", out)

def txt_from_docx(path: Path):
    doc = Document(path)
    text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    out = DATA_DIR / f"{path.stem}.txt"
    out.write_text(f"Source: {path.name}\n\n{text}", encoding='utf-8')
    print("wrote", out)

def main():
    for p in DATA_DIR.iterdir():
        if p.suffix.lower() in ('.xls', '.xlsx'):
            txt_from_xlsx(p)
        elif p.suffix.lower() == '.csv':
            txt_from_csv(p)
        elif p.suffix.lower() == '.docx':
            txt_from_docx(p)
        # leave PDFs alone (PyPDFLoader handles them)
    print("done")

if __name__ == "__main__":
    main()
