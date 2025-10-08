import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import subprocess
import shutil
from pathlib import Path


def epub_to_pdf(epub_path: Path, pdf_path: Path) -> bool:
    """Converts EPUB into PDF via Calibre (ebook-convert)."""
    calibre = shutil.which("ebook-convert")
    if not calibre:
        print("❌ Calibre (ebook-convert) not found in PATH or not installed.")
        return False

    try:
        cmd = [calibre, str(epub_path), str(pdf_path)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if pdf_path.exists():
            print(f"✅ EPUB converted via Calibre : {pdf_path}")
            return True
    except Exception as e:
        print(f"⚠️ Error converting EPUB with Calibre : {e}")

    return False


def _find_libreoffice() -> Optional[Path]:
    """Finds the path to LibreOffice/soffice if present, otherwise returns None."""
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
    """Converts a file into a persistant PDF in the same folder as the input file."""
    forbidden_ext = {".exe", ".zip", ".tar", ".gz", ".7z", ".bat", ".sh"}
    ext = input_file.suffix.lower()

    if ext in forbidden_ext:
        print(f"⚠️ File ignored for PDF conversion: {input_file}")
        return None

    output_pdf = input_file.with_suffix(".pdf")

    if ext == ".epub":
        calibre = shutil.which("ebook-convert")
        if calibre:
            try:
                cmd = [calibre, str(input_file), str(output_pdf)]
                subprocess.run(
                    cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                if output_pdf.exists():
                    print(f"✅ Converted via Calibre : {input_file}")
                    return output_pdf
            except Exception as e:
                print(f"⚠️ EPUB conversion failed with Calibre : {input_file} ({e})")

        # python fallback
        try:
            epub_to_pdf(input_file, output_pdf)
            if output_pdf.exists():
                return output_pdf
        except Exception as e:
            print(f"⚠️ Conversion EPUB→PDF failed (fallback) : {input_file} ({e})")
        return None

    # LibreOffice
    lo = _find_libreoffice()
    if lo:
        try:
            cmd = [
                str(lo),
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(input_file.parent),
                str(input_file),
            ]
            subprocess.run(
                cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            if output_pdf.exists():
                print(f"✅ Converted via LibreOffice : {input_file}")
                return output_pdf
        except Exception as e:
            print(f"⚠️ LibreOffice conversion failed : {input_file} ({e})")

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
                    print(f"✅ Converted via MS Word : {input_file}")
                    return output_pdf
            elif ext in {".xls", ".xlsx"}:
                excel = win32com.client.DispatchEx("Excel.Application")
                excel.Visible = False
                wb = excel.Workbooks.Open(str(input_file))
                wb.ExportAsFixedFormat(0, str(output_pdf))  # xlTypePDF
                wb.Close(False)
                excel.Quit()
                if output_pdf.exists():
                    print(f"✅ Converted via MS Excel : {input_file}")
                    return output_pdf
            elif ext in {".ppt", ".pptx"}:
                ppt = win32com.client.DispatchEx("PowerPoint.Application")
                pres = ppt.Presentations.Open(str(input_file), WithWindow=False)
                pres.SaveAs(str(output_pdf), 32)  # ppSaveAsPDF
                pres.Close()
                ppt.Quit()
                if output_pdf.exists():
                    print(f"✅ Converted via MS PowerPoint : {input_file}")
                    return output_pdf
        except Exception as e:
            print(f"⚠️ MS Office conversion failed : {input_file} ({e})")

    # Fallback docx2pdf
    if ext == ".docx":
        try:
            from docx2pdf import convert

            convert(str(input_file), str(output_pdf))
            if output_pdf.exists():
                print(f"✅ Converted via docx2pdf : {input_file}")
                return output_pdf
        except Exception as e:
            print(f"⚠️ docx2pdf conversion failed: {input_file} ({e})")

    # Fallback text → PDF
    if ext in {".txt", ".md", ".json", ".csv", ".yaml", ".yml", ".log"}:
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.units import inch
            from reportlab.pdfgen import canvas

            text = input_file.read_text(encoding="utf-8", errors="ignore")
            c = canvas.Canvas(str(output_pdf), pagesize=A4)
            width, height = A4
            margin = inch  # Using inches for better consistency
            y = height - margin

            # Using a more sophisticated text rendering approach
            text_object = c.beginText(margin, y)
            text_object.setFont("Courier", 10)  # Monospace font for code/text files
            for line in text.splitlines():
                if y < margin:
                    c.drawText(text_object)
                    c.showPage()
                    text_object = c.beginText(margin, height - margin)
                    y = height - margin

                # Handle long lines by splitting them
                if len(line) > 120:
                    for i in range(0, len(line), 120):
                        text_object.textLine(line[i : i + 120])
                else:
                    text_object.textLine(line)

            c.drawText(text_object)
            c.save()

            if output_pdf.exists():
                print(f"✅ Converted via reportlab : {input_file}")
                return output_pdf
        except Exception as e:
            print(f"⚠️ Text→PDF conversion failed : {input_file} ({e})")

    print(f"❌ Conversion impossible : {input_file}")
    return None
