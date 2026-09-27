# Intro Video Script — RAG Document Q&A

Target length: ~75-90 seconds. Screen recording + voiceover (record with
OBS Studio or Loom, both free). Tone: confident, plain-spoken, no hype
words ("revolutionary," "game-changing") — the project's whole pitch is
that it's *measured*, so the video should sound the same way.

Reference stills for every scene are in `docs/screenshots/video/` —
captured from the real running app with `scripts/capture_video_shots.py`,
so they show actual output, not mockups.

## Must-include checklist

Don't cut the video shorter without keeping these — they're the load-
bearing content, everything else is polish:

- [ ] **The problem statement** (Scene 2): most RAG demos aren't measured.
      This is the entire reason the project exists — cutting it turns the
      video into "yet another RAG demo," the exact thing it's not.
- [ ] **A real question + cited, expandable source chunk** (Scene 3):
      proves answers are grounded, not hallucinated — the citation *and*
      the expanded matching source text, not just the citation ID.
- [ ] **The eval harness actually running, with real numbers on screen**
      (Scene 4): Recall@3 / Precision@3 / MRR / Faithfulness tiles. This
      is the single most important shot in the video — it's the
      differentiator from every other RAG portfolio project.
- [ ] **One concrete, specific failure example** (Scene 4 voiceover): the
      "vacation in London" TF-IDF miss. A vague "it measures quality"
      claim is forgettable; a specific example of *catching a real
      failure* is what makes the eval harness credible.
- [ ] **The GitHub link, visible on screen** (Scene 6), not just spoken —
      viewers skim, they won't pause to write down a spoken URL.
- [ ] **Something that signals "this is real code"**: the green CI badge,
      the test count, or the Docker/CLI/API mention (Scene 5) — anything
      that distinguishes it from a notebook someone ran once.

Everything else (title card, music, exact wording) is negotiable — these
six are not.

---

## Scene 1 — Hook (0:00–0:10)

**Visual:** Black screen, then title card fades in: **"RAG Document Q&A"**
subtitle: **"...with retrieval you can actually measure"**

**Voiceover:**
> "Most RAG demos look impressive and tell you nothing about whether the
> retrieval is actually good. This one measures it."

---

## Scene 2 — The problem, fast (0:10–0:20)

**Visual:** Quick cut to the PRD or a simple text card: "Recall@k? MRR?
Faithfulness?" then a shrug emoji or crossed-out "vibes-based eval."

**Voiceover:**
> "Anyone can wire up an embeddings API and a prompt. Almost nobody
> brings a labeled test set and defends the numbers. That's the gap this
> project closes."

---

## Scene 3 — Live demo: asking a question (0:20–0:45)

**Visual:** Screen recording of the Streamlit app (`localhost:8501` or
the deployed Streamlit Cloud URL). Reference stills:
`docs/screenshots/video/scene3a_ask_empty.png` →
`scene3b_ask_typed.png` → `scene3c_ask_answered.png` →
`scene3d_ask_chunk_expanded.png`.
1. Show the "Ask a question" tab
2. Type: *"How many weeks of parental leave do I get?"*
3. Click **Ask**
4. Answer appears, citing `parental_leave_policy::0` at similarity 0.570
5. Expand the cited chunk to show the actual source text matches the answer

**Voiceover:**
> "Eighteen HR policy documents, chunked and embedded. Ask a question in
> plain English, and you get an answer with the exact source chunk it
> came from — not just a vague 'trust me.' Click to expand it, and you
> can see the answer really is in there."

**Note:** don't use a paraphrased/vocabulary-mismatch question here — the
default TF-IDF embedder is documented to fail on exactly that case (see
README). Save that nuance for Scene 4, where it's the actual point being
made, instead of undercutting this "it works" shot.

---

## Scene 4 — Live demo: the evaluation harness (0:45–1:05)

**Visual:** Switch to the "Evaluation results" tab. Click **Run
evaluation**. Metric tiles populate: Recall@3, Precision@3, MRR,
Faithfulness. Scroll to the per-question table. Reference stills:
`docs/screenshots/video/scene4a_eval_empty.png` →
`scene4b_eval_results.png`.

**Voiceover:**
> "And here's the actual point of the project: a fifty-question labeled
> set scores retrieval quality automatically — recall, mean reciprocal
> rank, and a faithfulness check that catches the model making things up.
> It's not just a scoreboard — it tells you *why* things fail. Ask 'how
> much vacation do I get if I'm based in London' and the TF-IDF baseline
> misses it, because the source document says 'leave' and 'UK,' never
> 'vacation' or 'London.' Swap in real sentence embeddings and it catches
> the same question — because they understand meaning, not just keywords."

---

## Scene 5 — Architecture flash (1:05–1:20)

**Visual:** Cut to the architecture diagram from the README (screenshot
or recreate as a simple animated diagram): loaders → chunking →
embeddings → store → retrieve/evaluate → generate → CLI/API.

**Voiceover:**
> "Everything's provider-agnostic and swappable by design — embeddings,
> vector store, LLM — with a CLI, a FastAPI endpoint, full test coverage,
> CI on every push, and it runs in Docker."

---

## Scene 6 — Close (1:20–1:30)

**Visual:** GitHub repo page, scrolled to show the title, the green
"tests passing" badge, and the pitch line all in one frame. Reference
still: `docs/screenshots/video/scene6_github_repo.png`. End card: repo
URL + "MIT licensed, free to run locally."

**Voiceover:**
> "Code's on GitHub, fully open source. Link's below."

**On-screen text (no voiceover):** `github.com/omiee1234/rag-document-qa`

---

## Recording notes

- **Screen resolution:** record at 1920x1080 even if your display is
  larger — crop/scale down, don't record at 4K and shrink (text gets
  soft).
- **Cursor:** move deliberately, pause half a second before each click
  so viewers can follow.
- **Pacing:** the timestamps above are targets, not hard cuts — better to
  run 100 seconds and feel unhurried than 75 seconds and feel rushed.
- **Music:** optional, low-volume instrumental under the voiceover only
  (YouTube Audio Library has free tracks with no attribution required).
- **Export:** 1080p, MP4, under 100MB if posting to LinkedIn directly
  (their native player performs better than a YouTube-link post).
- **Captions:** LinkedIn autoplays muted by default — burn in captions
  or add LinkedIn's auto-captions, or most viewers will never hear the
  voiceover at all.
