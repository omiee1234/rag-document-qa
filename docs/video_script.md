# Intro Video Script — RAG Document Q&A

Target length: ~75-90 seconds. Screen recording + voiceover (record with
OBS Studio or Loom, both free). Tone: confident, plain-spoken, no hype
words ("revolutionary," "game-changing") — the project's whole pitch is
that it's *measured*, so the video should sound the same way.

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
the deployed Streamlit Cloud URL).
1. Show the "Ask a question" tab
2. Type: *"How much vacation time do I get if I'm based in London?"*
3. Click **Ask**
4. Answer appears, citing `leave_policy_uk::0`
5. Expand the cited chunk to show the actual source text

**Voiceover:**
> "Eighteen HR policy documents, chunked and embedded. Ask a question in
> plain English — here, deliberately worded so it doesn't share any
> vocabulary with the source doc — and you get an answer with the exact
> chunk it came from, not just a vague 'trust me.'"

---

## Scene 4 — Live demo: the evaluation harness (0:45–1:05)

**Visual:** Switch to the "Evaluation results" tab. Click **Run
evaluation**. Metric tiles populate: Recall@3, Precision@3, MRR,
Faithfulness. Scroll to the per-question table.

**Voiceover:**
> "And here's the actual point of the project: a fifty-question labeled
> set scores retrieval quality automatically — recall, mean reciprocal
> rank, and a faithfulness check that catches the model making things up.
> Swap the embedding provider — TF-IDF, sentence-transformers, OpenAI —
> and you get a real, defensible before-and-after, not a guess."

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

**Visual:** GitHub repo page, scrolled to show the green CI badge and
star/fork buttons. End card: repo URL + "MIT licensed, free to run
locally."

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
