from ragqa.loaders import _strip_repeated_headers_footers


def _page(n, body):
    return "\n".join(
        [
            "Acme Insurance Ltd - Policy Schedule",
            *body,
            f"Page {n} of 5",
            "Registered office: 1 Main Street, Springfield",
        ]
    )


def test_repeated_header_and_footer_are_stripped_after_first_copy():
    pages = [_page(i, [f"Body clause {i} about coverage."]) for i in range(1, 6)]
    joined = "\n".join(_strip_repeated_headers_footers(pages))

    assert joined.count("Acme Insurance Ltd - Policy Schedule") == 1
    assert joined.count("Registered office: 1 Main Street") == 1
    # page numbers carry no information, so every one is removed
    assert not any(f"Page {i} of 5" in joined for i in range(1, 6))


def test_numbered_body_lines_are_not_merged_into_boilerplate():
    # Regression: an earlier version collapsed digits when comparing lines,
    # so "Clause 1 ..." / "Clause 2 ..." on short pages looked like one
    # repeated footer and got stripped.
    pages = ["\n".join([f"Clause {i}: the insurer covers item {i}."]) for i in range(1, 6)]
    joined = "\n".join(_strip_repeated_headers_footers(pages))
    for i in range(1, 6):
        assert f"Clause {i}: the insurer covers item {i}." in joined


def test_body_text_is_never_stripped():
    pages = [_page(i, [f"Body clause {i} about coverage."]) for i in range(1, 6)]
    joined = "\n".join(_strip_repeated_headers_footers(pages))
    for i in range(1, 6):
        assert f"Body clause {i} about coverage." in joined


def test_repeated_line_in_the_middle_of_a_page_is_kept():
    # a phrase that legitimately repeats in body text (outside the
    # header/footer zone) must survive every time
    long_body = [f"filler line {j} with enough length" for j in range(20)]
    pages = [
        _page(i, long_body[:10] + ["The sum insured is restored once per year."] + long_body[10:])
        for i in range(1, 6)
    ]
    joined = "\n".join(_strip_repeated_headers_footers(pages))
    assert joined.count("The sum insured is restored once per year.") == 5


def test_short_generic_lines_are_not_treated_as_boilerplate():
    pages = ["\n".join(["N/A", f"Answer {i} text", "Yes"]) for i in range(5)]
    joined = "\n".join(_strip_repeated_headers_footers(pages))
    assert joined.count("N/A") == 5
    assert joined.count("Yes") == 5


def test_header_on_consecutive_pages_is_stripped_even_if_rare_overall():
    # A bundled PDF: the last sub-document repeats its own header on 3
    # straight pages -- under 25% of all pages, but clearly a running header.
    body = [f"distinct body sentence number {i} here" for i in range(20)]
    pages = [f"Unrelated page {i}\n" + "\n".join(body[i:i + 1]) for i in range(10)]
    pages += [f"Acme Sub-Document Header\n{b}" for b in body[10:13]]
    joined = "\n".join(_strip_repeated_headers_footers(pages))
    assert joined.count("Acme Sub-Document Header") == 1


def test_documents_under_three_pages_are_left_alone():
    pages = [_page(1, ["One."]), _page(2, ["Two."])]
    assert _strip_repeated_headers_footers(pages) == pages
