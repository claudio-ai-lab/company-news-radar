from newsradar.markup import balance_markers, render_inline, strip_inline


def test_render_inline_converts_bold_and_italic() -> None:
    html = render_inline("Prices rose **6%**, a *preliminary* figure.")
    assert html == "Prices rose <strong>6%</strong>, a <em>preliminary</em> figure."


def test_render_inline_escapes_untrusted_html() -> None:
    html = render_inline('**<img src=x onerror="alert(1)">** <script>steal()</script>')
    assert "<script>" not in html
    assert "<img" not in html
    assert html.startswith("<strong>&lt;img")


def test_dangling_markers_are_removed() -> None:
    assert balance_markers("Cut at **six") == "Cut at six"
    assert "*" not in render_inline("an *unfinished thought")


def test_strip_inline() -> None:
    assert strip_inline("**Bold** and *italic* text") == "Bold and italic text"
    assert strip_inline(None) == ""


def test_arithmetic_asterisks_are_not_italic() -> None:
    assert "<em>" not in render_inline("2*3 equals 6 and 4*5 equals 20")
