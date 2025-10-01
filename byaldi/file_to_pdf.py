import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional
import subprocess
import shutil
from pathlib import Path

def epub_to_pdf(epub_path: Path, pdf_path: Path) -> bool:
    """Convertit un EPUB en PDF via Calibre (ebook-convert)."""
    calibre = shutil.which("ebook-convert")
    if not calibre:
        print("❌ Calibre (ebook-convert) n'est pas installé ou pas dans le PATH.")
        return False

    try:
        cmd = [calibre, str(epub_path), str(pdf_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if pdf_path.exists():
            print(f"✅ EPUB converti via Calibre : {pdf_path}")
            return True
    except Exception as e:
        print(f"⚠️ Erreur conversion EPUB avec Calibre : {e}")
    
    return False

def _find_libreoffice() -> Optional[Path]:
    """Retourne le chemin de LibreOffice/soffice si présent, sinon None."""
    candidates = [
        os.environ.get("LIBREOFFICE_PATH"),
        shutil.which("soffice"),
        shutil.which("libreoffice"),
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
        "/usr/bin/libreoffice",
        "/usr/bin/soffice",
        "/snap/bin/libreoffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


def _convert_to_pdf(input_file: Path) -> Optional[Path]:
    """Convertit un fichier en PDF persistant (dans le même dossier que l’input)."""
    forbidden_ext = {".exe", ".zip", ".tar", ".gz", ".7z", ".bat", ".sh"}
    ext = input_file.suffix.lower()

    if ext in forbidden_ext:
        print(f"⚠️ Fichier ignoré (non convertible en PDF) : {input_file}")
        return None

    # le PDF est sauvegardé dans le même dossier que l'input
    output_pdf = input_file.with_suffix(".pdf")

    # EPUB spécifique avec Calibre ou fallback ebooklib
    if ext == ".epub":
        calibre = shutil.which("ebook-convert")
        if calibre:
            try:
                cmd = [calibre, str(input_file), str(output_pdf)]
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if output_pdf.exists():
                    print(f"✅ Converti via Calibre : {input_file}")
                    return output_pdf
            except Exception as e:
                print(f"⚠️ Conversion EPUB échouée avec Calibre : {input_file} ({e})")

        # fallback Python pur
        try:
            epub_to_pdf(input_file, output_pdf)
            if output_pdf.exists():
                return output_pdf
        except Exception as e:
            print(f"⚠️ Conversion EPUB→PDF échouée (fallback) : {input_file} ({e})")
        return None

    # LibreOffice
    lo = _find_libreoffice()
    if lo:
        try:
            cmd = [str(lo), "--headless", "--convert-to", "pdf",
                   "--outdir", str(input_file.parent), str(input_file)]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if output_pdf.exists():
                print(f"✅ Converti via LibreOffice : {input_file}")
                return output_pdf
        except Exception as e:
            print(f"⚠️ Conversion LibreOffice échouée : {input_file} ({e})")

    # MS Office (Windows)
    if os.name == "nt":
        try:
            import win32com.client
            if ext in {".doc", ".docx", ".rtf"}:
                word = win32com.client.DispatchEx("Word.Application")
                word.Visible = False
                doc = word.Documents.Open(str(input_file))
                doc.SaveAs(str(output_pdf), FileFormat=17)  # wdFormatPDF
                doc.Close(False)
                word.Quit()
                if output_pdf.exists():
                    print(f"✅ Converti via MS Word : {input_file}")
                    return output_pdf
            elif ext in {".xls", ".xlsx"}:
                excel = win32com.client.DispatchEx("Excel.Application")
                excel.Visible = False
                wb = excel.Workbooks.Open(str(input_file))
                wb.ExportAsFixedFormat(0, str(output_pdf))  # xlTypePDF
                wb.Close(False)
                excel.Quit()
                if output_pdf.exists():
                    print(f"✅ Converti via MS Excel : {input_file}")
                    return output_pdf
            elif ext in {".ppt", ".pptx"}:
                ppt = win32com.client.DispatchEx("PowerPoint.Application")
                pres = ppt.Presentations.Open(str(input_file), WithWindow=False)
                pres.SaveAs(str(output_pdf), 32)  # ppSaveAsPDF
                pres.Close()
                ppt.Quit()
                if output_pdf.exists():
                    print(f"✅ Converti via MS PowerPoint : {input_file}")
                    return output_pdf
        except Exception as e:
            print(f"⚠️ Conversion MS Office échouée : {input_file} ({e})")

    # Fallback docx2pdf
    if ext == ".docx":
        try:
            from docx2pdf import convert
            convert(str(input_file), str(output_pdf))
            if output_pdf.exists():
                print(f"✅ Converti via docx2pdf : {input_file}")
                return output_pdf
        except Exception as e:
            print(f"⚠️ Conversion docx2pdf échouée : {input_file} ({e})")

    # Fallback texte → PDF
    if ext in {".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".log"}:
        try:
            text = input_file.read_text(encoding="utf-8", errors="ignore")
            c = canvas.Canvas(str(output_pdf), pagesize=A4)
            width, height = A4
            margin = 40
            y = height - margin
            for line in text.splitlines() or [""]:
                if y < margin:
                    c.showPage()
                    y = height - margin
                c.drawString(margin, y, line[:1200])
                y -= 14
            c.save()
            if output_pdf.exists():
                print(f"✅ Converti via reportlab : {input_file}")
                return output_pdf
        except Exception as e:
            print(f"⚠️ Conversion texte→PDF échouée : {input_file} ({e})")

    print(f"❌ Conversion impossible : {input_file}")
    return None
