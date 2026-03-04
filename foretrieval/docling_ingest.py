from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

def chunk_pdf_to_images(pdf_path: str, output_dir: str, scale: float = 5.0) -> List[Path]:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.transforms.chunker import HybridChunker
    from docling_core.types.doc.base import BoundingBox

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
    for item, level in doc.iterate_items():
        for prov in getattr(item, "prov", []):
            page_no = prov.page_no
            bbox = prov.bbox
            print(f"Len(item):{len(list(item.prov))} Type: {item.label}, Page: {page_no}, Level: {level}, BBox: {bbox}")
            page = doc.pages[page_no]
            if page.image:
                page_img = page.image.pil_image
                page_height = page.size.height
                bbox = prov.bbox.to_top_left_origin(page_height).normalized(page.size)
                bbox = [int(bbox.l * page_img.width), int(bbox.t * page_img.height), int(bbox.r * page_img.width), int(bbox.b * page_img.height)]
                output_path = output_dir / f"element_{item.label}_page{page_no}_level{level}_id{elem_count}.png"
                output_paths.append(output_path)
                cropped = draw_bbox(page_img, bbox)
                cropped.save(output_path)
                print(f"Saved cropped image to: {output_path.stem}")
                elem_count += 1
    
    chunker = HybridChunker(
        max_tokens=2048, # limite de tokens par chunk
        merge_peers=True # fusionne les petits chunks adjacents
    )

    for chunk in chunker.chunk(doc):    
        print(f"Headings: {chunk.meta.headings}") # Titres en texte
        # Grouper par page --> TODO: pourquoi ne pas regrouper des éléments de différentes pages
        # TODO: images / table disparues
        # TODO: pourquoi ne pas regrouper les captions avec les figures/tables?
        bboxes_by_page = {}
        for item in chunk.meta.doc_items:
            label = item.label
            has_bbox = any(prov.bbox for prov in getattr(item, "prov", []))
            print(f" - {label}: has_bbox={has_bbox}")

            for prov in getattr(item, "prov", []):
                if prov.bbox:
                    bboxes_by_page.setdefault(prov.page_no, []).append(prov.bbox)

        for page_no, bboxes in bboxes_by_page.items():
            page = doc.pages[page_no]
            page_img = page.image.pil_image

            # BBOX englobante convertie en coordonnées pixels
            enclosing_crop_box = BoundingBox.enclosing_bbox(bboxes)
            enc_tlo = enclosing_crop_box.to_top_left_origin(page.size.height).normalized(page.size)
            enc_box = (int(enc_tlo.l * page_img.width), int(enc_tlo.t * page_img.height),
                    int(enc_tlo.r * page_img.width), int(enc_tlo.b * page_img.height))

            # Masquage
            mask = Image.new("RGB", page_img.size, "white")
            output_path = output_dir / f"mask_chunked_element_{item.label}_page{page_no}_id{elem_count}_nbbox{len(bboxes)}.png"
            for bbox in bboxes:
                tlo = bbox.to_top_left_origin(page.size.height).normalized(page.size)
                box = (int(tlo.l * page_img.width), int(tlo.t * page_img.height),
                    int(tlo.r * page_img.width), int(tlo.b * page_img.height))
                region = page_img.crop(box)
                mask.paste(region, box)
            mask.crop(enc_box).save(output_path)
            output_paths.append(output_path)
    
            output_path = output_dir / f"chunked_element_{item.label}_page{page_no}_id{elem_count}_nbbox{len(bboxes)}.png"
            cropped = draw_bbox(page_img, enc_box)
            cropped.save(output_path)
            output_paths.append(output_path)
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