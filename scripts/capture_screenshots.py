"""One-off script to capture README screenshots of the running Streamlit app.

Not part of the package; run manually with the app already up:
    streamlit run src/ragqa/ui.py
    python scripts/capture_screenshots.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
APP_URL = "http://localhost:8501"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 700})
        page.goto(APP_URL)
        page.wait_for_selector("text=Ask a question", timeout=15000)
        page.wait_for_timeout(1000)

        # --- Ask a question tab ---
        question_box = page.get_by_placeholder("How many days of PTO do I get?")
        question_box.click()
        question_box.fill("How many weeks of parental leave do I get?")
        page.get_by_role("button", name="Ask").click()
        page.wait_for_selector("text=Answer", timeout=15000)
        page.wait_for_selector("text=Cited chunks", timeout=15000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT_DIR / "ask.png"), full_page=True)
        print(f"Saved {OUT_DIR / 'ask.png'}")

        # --- Evaluation results tab ---
        page.get_by_role("tab", name="Evaluation results").click()
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Run evaluation").click()
        page.wait_for_selector("text=Per-question detail", timeout=30000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT_DIR / "eval.png"), full_page=True)
        print(f"Saved {OUT_DIR / 'eval.png'}")

        browser.close()


if __name__ == "__main__":
    main()
