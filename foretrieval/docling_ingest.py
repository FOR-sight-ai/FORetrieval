from pathlib import Path
import hashlib
from typing import Dict, Iterable, List, Optional, Tuple, NamedTuple
from PIL import Image

class PendingImg(NamedTuple):
    page_no: int
    y_top: float     # pour trier dans la page
    img: Image.Image

class ExportedImg(NamedTuple):
    path: Path
    page_id: int
    elem_id: int

# -----------------------------
# Geometry helpers
# -----------------------------
def bbox_to_norm_ltrb(bbox, page) -> Tuple[float, float, float, float]:
    bb = bbox.to_top_left_origin(page.size.height).normalized(page.size)
    return (bb.l, bb.t, bb.r, bb.b)

def norm_ltrb_to_px(ltrb: Tuple[float, float, float, float], page_img: Image.Image) -> Tuple[int, int, int, int]:
    l, t, r, b = ltrb
    return (int(l * page_img.width), int(t * page_img.height),
            int(r * page_img.width), int(b * page_img.height))

def area_inclusion(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    l = max(a[0], b[0])
    t = max(a[1], b[1])
    r = min(a[2], b[2])
    bb = min(a[3], b[3])

    inter_w = max(0.0, r - l)
    inter_h = max(0.0, bb - t)
    inter = inter_w * inter_h

    a_w = max(0.0, a[2] - a[0])
    a_h = max(0.0, a[3] - a[1])
    area_a = a_w * a_h

    return 0.0 if area_a <= 0 else inter / area_a


# -----------------------------
# Lightweight indexes
# -----------------------------
def build_picture_index(doc) -> List[dict]:
    pics = []
    for it, _lvl in doc.iterate_items():
        if getattr(it, "label", None) != "picture":
            continue
        for prov in getattr(it, "prov", []):
            if not getattr(prov, "bbox", None):
                continue
            page_no = prov.page_no
            page = doc.pages[page_no]
            pics.append({
                "self_ref": getattr(it, "self_ref", None),
                "page_no": page_no,
                "bbox_norm": bbox_to_norm_ltrb(prov.bbox, page),
            })
    return pics

def build_heading_text_index(doc) -> List[dict]:
    HEADING_LABELS = {"title", "heading", "section_header", "header", "subtitle"}
    headings = []
    for it, _lvl in doc.iterate_items():
        if getattr(it, "label", None) not in HEADING_LABELS:
            continue
        text = getattr(it, "text", None) or getattr(it, "value", None) or getattr(it, "content", None)
        for prov in getattr(it, "prov", []):
            if not getattr(prov, "bbox", None):
                continue
            page_no = prov.page_no
            page = doc.pages[page_no]
            headings.append({
                "text": text,
                "page_no": page_no,
                "bbox_norm": bbox_to_norm_ltrb(prov.bbox, page),
            })
    return headings

def build_caption_index(doc) -> List[dict]:
    CAPTION_LABELS = {"caption", "figure_caption", "table_caption"}
    out = []
    for it, _lvl in doc.iterate_items():
        if getattr(it, "label", None) not in CAPTION_LABELS:
            continue
        txt = getattr(it, "text", None) or getattr(it, "value", None) or getattr(it, "content", None)
        for prov in getattr(it, "prov", []):
            if not getattr(prov, "bbox", None):
                continue
            page_no = prov.page_no
            page = doc.pages[page_no]
            out.append({
                "page_no": page_no,
                "bbox_norm": bbox_to_norm_ltrb(prov.bbox, page),
                "text": txt,
            })
    return out

def image_fingerprint(img: Image.Image, thumb: int = 512) -> str:
    # Fingerprint stable: convert, resize to limit cost, then hash bytes
    x = img.convert("RGB")
    x.thumbnail((thumb, thumb))
    h = hashlib.blake2b(x.tobytes(), digest_size=16)
    return h.hexdigest()

# -----------------------------
# Heading matching (ton code)
# -----------------------------
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return " ".join(str(s).strip().lower().split())


def find_heading_item_for_chunk(chunk, headings_index: List[dict], lookback_pages: int = 3) -> Optional[dict]:
    if not getattr(chunk.meta, "headings", None):
        return None
    target = normalize_text(chunk.meta.headings[-1])
    if not target:
        return None

    chunk_pages = set()
    for it in chunk.meta.doc_items:
        for prov in getattr(it, "prov", []):
            if getattr(prov, "bbox", None):
                chunk_pages.add(prov.page_no)

    def text_match(h_text: str) -> bool:
        ht = normalize_text(h_text)
        return bool(ht) and (ht == target or target in ht or ht in target)

    if not chunk_pages:
        candidates = [h for h in headings_index if text_match(h.get("text"))]
        if not candidates:
            return None
        candidates.sort(key=lambda x: (x["page_no"], x["bbox_norm"][3]), reverse=True)
        return candidates[0]

    min_page = min(chunk_pages)
    allowed_pages = set(chunk_pages)
    for k in range(1, lookback_pages + 1):
        p = min_page - k
        if p >= 0:
            allowed_pages.add(p)

    candidates = []
    for h in headings_index:
        if h["page_no"] not in allowed_pages:
            continue
        if not text_match(h.get("text")):
            continue
        dist = 0 if h["page_no"] in chunk_pages else (min_page - h["page_no"])
        candidates.append((dist, h))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[1]["page_no"], -x[1]["bbox_norm"][3]))
    return candidates[0][1]


# -----------------------------
# Minimal rendering helpers
# -----------------------------
def enclosing_bbox_norm(boxes: Iterable[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))

def stack_vertical(top: Image.Image, bottom: Image.Image, gap: int = 6) -> Image.Image:
    w = max(top.width, bottom.width)
    h = top.height + gap + bottom.height
    out = Image.new("RGB", (w, h), "white")
    out.paste(top, (0, 0))
    out.paste(bottom, (0, top.height + gap))
    return out

def best_caption_for_picture(pic: dict, captions: List[dict], max_gap: float = 0.06, min_h_overlap: float = 0.5) -> Optional[dict]:
    pno = pic["page_no"]
    p_l, p_t, p_r, p_b = pic["bbox_norm"]
    best, best_score = None, -1.0

    for cap in captions:
        if cap["page_no"] != pno:
            continue
        c_l, c_t, c_r, c_b = cap["bbox_norm"]
        if c_t < p_b:
            continue
        gap = c_t - p_b
        if gap > max_gap:
            continue

        inter_w = max(0.0, min(p_r, c_r) - max(p_l, c_l))
        p_w = max(1e-6, (p_r - p_l))
        overlap = inter_w / p_w
        if overlap < min_h_overlap:
            continue

        score = (1.0 - (gap / max_gap)) * overlap
        if score > best_score:
            best_score, best = score, cap

    return best


# -----------------------------
# Export orphans (seul export "secondaire")
# -----------------------------
def export_orphan_pictures_pending(doc,
                                   pictures: List[dict],
                                   captions: List[dict],
                                   included: set,
                                   gap: int = 6) -> List[Tuple[int, float, Image.Image]]:
    pending = []
    for pic in pictures:
        key = (pic["page_no"], tuple(round(x, 4) for x in pic["bbox_norm"]))
        if key in included:
            continue

        page_no = pic["page_no"]
        page_img = doc.pages[page_no].image.pil_image

        pic_crop = page_img.crop(norm_ltrb_to_px(pic["bbox_norm"], page_img))

        cap = best_caption_for_picture(pic, captions)
        if cap is not None:
            cap_crop = page_img.crop(norm_ltrb_to_px(cap["bbox_norm"], page_img))
            out_img = stack_vertical(pic_crop, cap_crop, gap=gap)
        else:
            out_img = pic_crop

        # y_top = top de la picture => ordre naturel
        y_top = pic["bbox_norm"][1]
        pending.append((page_no, y_top, out_img))

    return pending


# -----------------------------
# MAIN: chunk_pdf_to_images (structuré)
# -----------------------------
def chunk_pdf_to_images(pdf_path: str, output_dir: str, scale: float = 2.0, max_tokens: int = 512) -> List[ExportedImg]:
    """
    Convertit un PDF en images "chunks" (texte+images) + images orphelines (pictures non capturées).
    Paramètres:
        - pdf_path: chemin vers le PDF d'entrée
        - output_dir: dossier de sortie où seront créés des sous-dossiers par document
        - scale: facteur de mise à l'échelle pour les images extraites (par exemple, 2.0 pour doubler la résolution)
        - max_tokens: nombre maximum de tokens par chunk (utilisé pour le chunking du texte)

    Sortie:
      output_dir/<doc_stem>/
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling_core.transforms.chunker import HybridChunker

    AREA_THRESH = 0.10 # seuil d'inclusion d'une picture dans un chunk (en fonction de la proportion de sa surface incluse dans le chunk)
    GAP = 15 # gap en pixels entre picture et caption lors du stacking vertical

    pdf_path = str(pdf_path)
    doc_stem = Path(pdf_path).stem

    # (A) Préparer dossier de sortie
    out_images_dir = Path(output_dir)
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # (B) Convertir PDF -> docling document (avec images pages)
    pipeline_options = PdfPipelineOptions(
        generate_page_images=True,
        generate_picture_images=True,
        images_scale=scale,
    )
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
    chunker = HybridChunker(max_tokens=max_tokens, merge_peers=True)

    doc = converter.convert(pdf_path).document

    # (C) Indexes (une fois)
    pictures = build_picture_index(doc)
    headings = build_heading_text_index(doc)
    captions = build_caption_index(doc)

    # index pictures par page pour éviter de filtrer à chaque fois
    pics_by_page: Dict[int, List[dict]] = {}
    for p in pictures:
        pics_by_page.setdefault(p["page_no"], []).append(p)

    chunks = list(chunker.chunk(doc))

    # (D) Export chunks
    #     - included = set des pictures déjà incluses dans un chunk
    included_pic_keys = set()
    out_items: List[ExportedImg] = []
    pending: List[PendingImg] = []

    for chunk_id, chunk in enumerate(chunks):
        header_item = find_heading_item_for_chunk(chunk, headings)

        # bboxes du chunk par page (norm)
        boxes_by_page: Dict[int, List[Tuple[float, float, float, float]]] = {}
        for item in chunk.meta.doc_items:
            for prov in getattr(item, "prov", []):
                if not getattr(prov, "bbox", None):
                    continue
                pno = prov.page_no
                page = doc.pages[pno]
                boxes_by_page.setdefault(pno, []).append(bbox_to_norm_ltrb(prov.bbox, page))

        if not boxes_by_page:
            continue

        for page_no, item_boxes_norm in boxes_by_page.items():
            page_img = doc.pages[page_no].image.pil_image

            # enclosing bbox = crop final
            chunk_bb_norm = enclosing_bbox_norm(item_boxes_norm)
            enc_px = norm_ltrb_to_px(chunk_bb_norm, page_img)

            # construire la liste des "regions" à garder (items + pictures incluses)
            regions_px: List[Tuple[int, int, int, int]] = [
                norm_ltrb_to_px(b, page_img) for b in item_boxes_norm
            ]

            for pic in pics_by_page.get(page_no, []):
                if area_inclusion(pic["bbox_norm"], chunk_bb_norm) >= AREA_THRESH:
                    regions_px.append(norm_ltrb_to_px(pic["bbox_norm"], page_img))
                    included_pic_keys.add((pic["page_no"], tuple(round(x, 4) for x in pic["bbox_norm"])))

            # rendre le chunk masqué puis crop
            mask = Image.new("RGB", page_img.size, "white")
            for box in regions_px:
                mask.paste(page_img.crop(box), box)
            chunk_crop = mask.crop(enc_px)

            # préfixer header si trouvé
            if header_item is not None:
                header_img = doc.pages[header_item["page_no"]].image.pil_image
                header_crop = header_img.crop(norm_ltrb_to_px(header_item["bbox_norm"], header_img))
                out_img = stack_vertical(header_crop, chunk_crop, gap=GAP)
            else:
                out_img = chunk_crop

            y_top = chunk_bb_norm[1]  # top normalisé du chunk sur la page
            pending.append(PendingImg(page_no=page_no, y_top=y_top, img=out_img))

    # (E) Sauver les pictures orphelines (non incluses dans les chunks)
    pending_orphans = export_orphan_pictures_pending(
        doc=doc,
        pictures=pictures,
        captions=captions,
        included=included_pic_keys,
        gap=GAP,
    )
    for page_no, y_top, img in pending_orphans:
        pending.append(PendingImg(page_no=page_no, y_top=y_top, img=img))

    # (F) Tri + retirer les doublons + écriture finale (ordre pages)
    pending.sort(key=lambda x: (x.page_no, x.y_top))

    seen_fps = set()
    elem_id = 0
    for it in pending:
        fp = image_fingerprint(it.img)
        if fp in seen_fps:
            continue
        seen_fps.add(fp)

        elem_id += 1
        out_path = out_images_dir / f"{doc_stem}_p{it.page_no:03d}_{elem_id:06d}.png"
        it.img.save(out_path)

        out_items.append(ExportedImg(
            path=out_path,
            page_id=it.page_no,
            elem_id=elem_id,
        ))

    return out_items


# ---------------------------------------------------
# MAIN TEST
# ---------------------------------------------------
if __name__ == "__main__":
    pdfs = ["data/doc_arduino.pdf", "data/doc_cnes.pdf", "data/doc_cvpr.pdf"]
    out_root = "chunks_output"

    for pdf_path in pdfs:
        paths = chunk_pdf_to_images(pdf_path, out_root, scale=5.0)
        print(f"✅ {len(paths)} images générées pour {pdf_path}")