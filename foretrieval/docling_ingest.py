from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

def chunk_pdf_to_images(pdf_path: str, output_dir: str, scale: float = 5.0) -> List[Path]:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.types.doc import ContentLayer
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_options = PdfPipelineOptions(
        generate_page_images=True, # images des pages complètes
        generate_picture_images=True, # images des figures détectées
        images_scale=scale # résolution (augmenter pour meilleure qualité)
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    result = converter.convert(pdf_path)
    doc = result.document
    output_paths = []
    elem_count = 0

    # Accéder aux éléments
    for item, level in doc.iterate_items(
        included_content_layers={ContentLayer.BODY, ContentLayer.FURNITURE}
    ):
        for prov in getattr(item, "prov", []):
            page_no = prov.page_no
            bbox = prov.bbox
            print(f"Type: {item.label}, Page: {page_no}, Level: {level}, BBox: {bbox}")
            page = doc.pages[page_no]
            if page.image:
                page_img = page.image.pil_image
                page_height = page.size.height
                bbox = prov.bbox.to_top_left_origin(page_height).normalized(page.size)
                bbox = [int(bbox.l * page_img.width), int(bbox.t * page_img.height), int(bbox.r * page_img.width), int(bbox.b * page_img.height)]
                cropped = draw_bbox(page_img, bbox)
                output_path = output_dir / f"element_{item.label}_page{page_no}_level{level}_id{elem_count}.png"
                cropped.save(output_path)
                output_paths.append(output_path)
                print(f"Saved cropped image to: {output_path.stem}")
                elem_count += 1
    return output_paths

def draw_bbox(page_img, bbox, outline="red", width=4):
    img = page_img.copy()
    d = ImageDraw.Draw(img)
    d.rectangle(bbox, outline=outline, width=width)
    return img

# ---------------------------------------------------
# MAIN TEST
# ---------------------------------------------------
if __name__ == "__main__":
    pdf = "data/doc_cvpr.pdf"          # remplace par ton fichier
    out_dir = "chunks_output"
    paths = chunk_pdf_to_images(pdf, out_dir)
    print(f"\n✅ {len(paths)} images générées :")
    for p in paths:
        print(p)