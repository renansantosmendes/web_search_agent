"""Tests of the markdown cleaner used for the cleaned_content field."""

import pytest

from tools.content_cleaner import clean_markdown


def test_images_are_removed():
    assert clean_markdown("Antes ![foto](https://x.com/a.jpg) depois") == (
        "Antes depois"
    )


def test_linked_images_are_removed_with_their_link():
    text = '[![logo](https://x.com/logo.svg "Portal")](https://x.com)'

    assert clean_markdown(text) == ""


def test_links_keep_their_text():
    text = "Leia o [relatório completo](https://x.com/relatorio) agora"

    assert clean_markdown(text) == "Leia o relatório completo agora"


def test_navigation_links_to_sections_are_dropped():
    text = "[Economia](/economia) [Dólar sobe](/economia/noticia/dolar)"

    assert clean_markdown(text) == "Dólar sobe"


def test_extra_blank_lines_are_collapsed_into_one():
    text = "primeiro\n\n\n\n   \nsegundo"

    assert clean_markdown(text) == "primeiro\n\nsegundo"


def test_whitespace_inside_lines_is_normalized():
    assert clean_markdown("  um    texto\tcom   espaços  ") == "um texto com espaços"


@pytest.mark.parametrize("text", ["", "   \n\n  ", "![so imagem](x.png)"])
def test_text_without_content_returns_empty_string(text):
    assert clean_markdown(text) == ""


def test_news_page_markup_is_removed_and_text_is_kept():
    text = (
        "[Pular para o conteúdo principal](#main-content)\n\n"
        '[![logo ebc](https://x.com/logo.webp)](https://ebc.com.br "Portal")\n\n'
        "# Economia\n\n"
        "[Economia](/economia) [# Importações para pesquisa pública\n\n"
        "Universidades poderão destinar recursos.](/economia/noticia/2026-10/a)\n\n"
        '[![Dólar](/sites/default/files/x.gif "Valter")](/economia/noticia/dolar)\n\n'
        "[Economia](/economia) [Dólar cai para R$ 5,01](/economia/noticia/dolar)\n\n"
        "ter, 20/10/2026 - 19:34\n\n"
        "* [Ver mais ![seta](https://x.com/arrow.svg)](/economia?page=1)"
    )

    cleaned = clean_markdown(text)

    assert "![" not in cleaned
    assert "](" not in cleaned
    assert "Dólar cai para R$ 5,01" in cleaned
    assert "Universidades poderão destinar recursos." in cleaned
    assert "ter, 20/10/2026 - 19:34" in cleaned
