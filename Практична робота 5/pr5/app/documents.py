"""Колекція документів: читання, метадані, поділ на фрагменти.

Це єдине місце, яке знає, як влаштовані файли в `docs/`: де в них
метадані, як розмічено текст, за якими межами його ділити. Решта
застосунку працює з готовими фрагментами (`Chunk`) і не читає файлів.

Розбір блоку метаданих реалізовано: це формат файлів, а не предмет
роботи. Поділ на фрагменти — заготовка: розмір, межі, перекриття і те,
що саме потрапляє в кожен фрагмент, — рішення з розділу 2 практичної
роботи.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Параметри поділу — відправна точка, а не рекомендація. У чому їх
# рахувати (символи, слова, токени моделі) і як застосовувати, коли
# ділите за заголовками, — ваше рішення.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс те, що ви вирішите додати самі:
    заголовок розділу, порядковий номер фрагмента, позицію в документі.
    Фільтри пошуку працюють саме з цим словником.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку
    файлу. Повертає словник метаданих і решту тексту. Порожні значення
    (`product:` без нічого) стають порожнім рядком. Якщо блоку немає —
    порожній словник і текст як є.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Поділити документ на фрагменти за заголовками й абзацами.

    Спочатку документ ділиться на секції за заголовками другого рівня.
    Назва документа та заголовок секції додаються до тексту фрагмента,
    щоб окремий уривок залишався зрозумілим без сусідніх фрагментів.

    Якщо секція завелика, вона додатково ділиться на перекривні
    фрагменти приблизно заданого розміру.
    """
    import re

    if not text.strip():
        return []

    title = metadata.get("title", source)

    # Розділяємо документ на секції за заголовками ##.
    sections = re.split(r"(?m)^##\s+", text.strip())

    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        section = section.strip()

        if not section:
            continue

        lines = section.splitlines()
        section_title = lines[0].strip()
        section_body = "\n".join(lines[1:]).strip()

        if not section_body:
            continue

        # Заголовок і назва документа завжди є частиною embedding-тексту.
        prefix = f"Документ: {title}\nРозділ: {section_title}\n\n"

        # Спочатку намагаємося залишати абзаци цілими.
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", section_body)
            if paragraph.strip()
        ]

        pieces: list[str] = []
        current = ""

        for paragraph in paragraphs:
            candidate = paragraph if not current else current + "\n\n" + paragraph

            if len(candidate) <= CHUNK_SIZE:
                current = candidate
            else:
                if current:
                    pieces.append(current)

                # Один окремий абзац теж може бути довшим за CHUNK_SIZE.
                if len(paragraph) > CHUNK_SIZE:
                    start = 0
                    while start < len(paragraph):
                        end = start + CHUNK_SIZE
                        pieces.append(paragraph[start:end])

                        if end >= len(paragraph):
                            break

                        start = max(end - CHUNK_OVERLAP, start + 1)

                    current = ""
                else:
                    current = paragraph

        if current:
            pieces.append(current)

        # Якщо секція взагалі не розбилась, все одно створюємо один chunk.
        if not pieces:
            pieces = [section_body]

        # Додаємо overlap між великими частинами секції.
        for piece_index, piece in enumerate(pieces):
            piece = piece.strip()

            if not piece:
                continue

            chunk_text = prefix + piece

            chunk_metadata = dict(metadata)
            chunk_metadata["section"] = section_title
            chunk_metadata["chunk_index"] = chunk_index
            chunk_metadata["section_chunk_index"] = piece_index

            chunks.append(
                Chunk(
                    text=chunk_text,
                    source=source,
                    metadata=chunk_metadata,
                )
            )

            chunk_index += 1

    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
