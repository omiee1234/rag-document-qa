"""Capture reference screenshots for each scene of docs/video_script.md.

Run with the app already up:
    streamlit run src/ragqa/ui.py
    python scripts/capture_video_shots.py
"""

from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "screenshots" / "video"
OUT_DIR.mkdir(parents=True, exist_ok=True)
APP_URL = "http://localhost:8501"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        page.goto(APP_URL)
        page.wait_for_selector("text=Ask a question", timeout=15000)
        page.wait_for_timeout(1000)

        # Scene 3a: empty question box, ready to type (framing shot)
        page.screenshot(path=str(OUT_DIR / "scene3a_ask_empty.png"))
        print("Saved scene3a_ask_empty.png")

        # Scene 3b: question typed, before clicking Ask
        question_box = page.get_by_placeholder("How many days of PTO do I get?")
        question_box.click()
        question_box.fill("How many weeks of parental leave do I get?")
        page.screenshot(path=str(OUT_DIR / "scene3b_ask_typed.png"))
        print("Saved scene3b_ask_typed.png")

        # Scene 3c: answer + citations after clicking Ask
        page.get_by_role("button", name="Ask").click()
        page.wait_for_selector("text=Cited chunks", timeout=15000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT_DIR / "scene3c_ask_answered.png"), full_page=True)
        print("Saved scene3c_ask_answered.png")

        # Scene 3d: expanded cited chunk showing source text
        expanders = page.locator('[data-testid="stExpander"]')
        expanders.first.click()
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT_DIR / "scene3d_ask_chunk_expanded.png"), full_page=True)
        print("Saved scene3d_ask_chunk_expanded.png")

        # Scene 4a: evaluation tab before running
        page.get_by_role("tab", name="Evaluation results").click()
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT_DIR / "scene4a_eval_empty.png"))
        print("Saved scene4a_eval_empty.png")

        # Scene 4b: metrics + per-question table after running
        page.get_by_role("button", name="Run evaluation").click()
        page.wait_for_selector("text=Per-question detail", timeout=30000)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT_DIR / "scene4b_eval_results.png"), full_page=True)
        print("Saved scene4b_eval_results.png")

        browser.close()


if __name__ == "__main__":
    main()
