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
        page = browser.new_page(viewport={"width": 1100, "height": 900})
        page.goto(APP_URL)
        # first run can load the hybrid models (~1 min)
        page.wait_for_selector("text=Document source", timeout=300000)
        page.wait_for_timeout(1000)

        # --- Ask a question tab ---
        question_box = page.get_by_placeholder("How many days of PTO do I get?")
        question_box.click()
        question_box.fill("How much vacation time do I get if I'm based in London?")
        question_box.press("Enter")
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Ask").click()
        page.wait_for_selector("text=Sources", timeout=120000)
        page.wait_for_timeout(2000)
        page.screenshot(path=str(OUT_DIR / "ask.png"), full_page=True)
        print(f"Saved {OUT_DIR / 'ask.png'}")

        # --- Evaluation results tab ---
        page.get_by_role("tab", name="Evaluation results").click()
        page.wait_for_timeout(500)
        page.get_by_role("button", name="Run evaluation").click()
        page.wait_for_selector("text=Per-question detail", timeout=600000)
        page.wait_for_timeout(1000)
        page.screenshot(path=str(OUT_DIR / "eval.png"), full_page=True)
        print(f"Saved {OUT_DIR / 'eval.png'}")

        browser.close()


if __name__ == "__main__":
    main()
