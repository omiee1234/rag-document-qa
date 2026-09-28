from ragqa.chunking import chunk_structured
from ragqa.embeddings import tfidf_analyzer
from ragqa.loaders import _heading_shaped, _table_blocks
from ragqa.structure import HEADING, PARAGRAPH, TABLE_ROW, Block, StructuredDoc, parse_markdown


# --- markdown parsing -------------------------------------------------------

def test_markdown_title_headings_and_paragraphs():
    doc = parse_markdown(
        "hr",
        "# Leave Policy\n\nEmployees get 18 days\nof paid leave.\n\n## Carryover\n\nUp to 5 days carry over.\n",
    )
    assert doc.title == "Leave Policy"
    assert [(b.kind, b.text) for b in doc.blocks] == [
        (PARAGRAPH, "Employees get 18 days of paid leave."),
        (HEADING, "Carryover"),
        (PARAGRAPH, "Up to 5 days carry over."),
    ]


def test_plain_text_without_headings_has_no_title():
    doc = parse_markdown("notes", "Just a paragraph.\n\nAnother one.")
    assert doc.title == ""
    assert len(doc.blocks) == 2


# --- structured chunking ----------------------------------------------------

def _doc():
    return StructuredDoc(
        doc_id="policy",
        title="Optima Secure",
        blocks=[
            Block("Free Look Period", HEADING, 2),
            Block("You may cancel within 30 days.", PARAGRAPH, 2),
            Block("Migration", HEADING, 3),
            Block("Apply 30 days before renewal.", PARAGRAPH, 3),
        ],
    )


def test_chunks_never_cross_a_heading_and_carry_section_and_page():
    chunks = chunk_structured(_doc())
    assert [(c.section, c.page, c.text) for c in chunks] == [
        ("Free Look Period", 2, "You may cancel within 30 days."),
        ("Migration", 3, "Apply 30 days before renewal."),
    ]
    assert [c.chunk_id for c in chunks] == ["policy::0", "policy::1"]


def test_embed_text_prefixes_context_but_answer_text_stays_verbatim():
    chunk = chunk_structured(_doc())[0]
    assert chunk.text == "You may cancel within 30 days."
    assert chunk.embed_text == "Optima Secure › Free Look Period\nYou may cancel within 30 days."


def test_blocks_are_packed_up_to_chunk_size():
    doc = StructuredDoc("d", blocks=[Block(f"Row {i} " + "x" * 40) for i in range(10)])
    chunks = chunk_structured(doc, chunk_size=200)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert sum(c.text.count("Row ") for c in chunks) == 10  # nothing lost or duplicated


def test_oversized_block_falls_back_to_character_chunking():
    doc = StructuredDoc("d", blocks=[Block("word " * 400)])
    chunks = chunk_structured(doc, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c.text) <= 300 + 10 for c in chunks)


# --- tables -----------------------------------------------------------------

def test_table_title_row_becomes_heading_and_header_row_labels_values():
    rows = [
        ["Insured Person Details", "", ""],
        ["Name", "Relationship", "Gender"],
        ["A. Sample", "Self", "Male"],
        ["B. Sample", "Wife", "Female"],
    ]
    blocks = _table_blocks(rows, page=3)
    assert blocks[0] == Block("Insured Person Details", HEADING, 3)
    assert blocks[1] == Block("Name: A. Sample; Relationship: Self; Gender: Male", TABLE_ROW, 3)
    assert blocks[2].text == "Name: B. Sample; Relationship: Wife; Gender: Female"


def test_only_first_multi_cell_row_can_be_a_header():
    # Regression: a later text-only row was taken as the header, attaching
    # the wrong labels to the values below it.
    rows = [
        ["Policy Number", "12345678", "Issuance Date", "16-09-2026"],
        ["Intermediary Name", "Intermediary Code", "Contact", "Email"],
        ["JOHN SAMPLE", "201651846566", "8511447426", "x@y.com"],
    ]
    texts = [b.text for b in _table_blocks(rows, page=1)]
    assert "JOHN SAMPLE | 201651846566 | 8511447426 | x@y.com" in texts
    assert not any("Intermediary Name: JOHN SAMPLE" in t for t in texts)


def test_wrapped_cell_text_is_merged_back_into_one_row():
    rows = [
        ["S.No", "Title", "Description"],
        ["3", "Type of", "Indemnity"],
        ["", "Insurance Product", ""],
    ]
    texts = [b.text for b in _table_blocks(rows, page=1)]
    assert texts == ["S.No: 3; Title: Type of Insurance Product; Description: Indemnity"]


def test_value_under_blank_header_cell_is_kept_not_dropped():
    # Regression: merged header cells shift columns; a value under a blank
    # header was silently dropped.
    rows = [
        ["S.No", "Title", "", "Description"],
        ["1", "Product", "Unlimited Restore", "Add-on cover"],
    ]
    texts = [b.text for b in _table_blocks(rows, page=1)]
    assert texts == ["S.No: 1; Title: Product; Unlimited Restore; Description: Add-on cover"]


# --- heading heuristics -----------------------------------------------------

def test_sentence_fragment_is_not_a_heading():
    assert _heading_shaped("Free Look Cancellation")
    assert not _heading_shaped("All exclusions applicable to the base product will")


def test_list_items_labels_and_fragments_are_not_headings():
    # Regression: a bold numbered list item became a section and mislabelled
    # the unrelated text that followed it.
    for text in [
        "2. Per Claim Deductible (Applicable for each and every claim",
        "a) Waiting period",
        "(iii) Pre-existing diseases",
        "Note:",
        "proposal form before buying a policy",
        "Some wrapped heading Non-",
        "Web-link: https://example.com/download",
    ]:
        assert not _heading_shaped(text), text
    assert _heading_shaped("Things to remember")


def test_label_with_long_code_is_not_a_heading():
    # Regression: "Application No <code>" became a heading, hiding the value
    # in section metadata instead of the searchable chunk text.
    assert not _heading_shaped("Application No HE01862691Q202609")
    assert _heading_shaped("Section D.1.9")


# --- tokenization -----------------------------------------------------------

def test_plural_and_singular_tokenize_the_same():
    assert tfidf_analyzer("insured persons policies") == tfidf_analyzer("insured person policy")
    assert "the" not in tfidf_analyzer("the policy")
