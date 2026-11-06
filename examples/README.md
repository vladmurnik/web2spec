# Examples

These examples are intentionally small. They need the runtime models; see the
main [README](../README.md).

## Collect facts from known URLs

`task.txt`:

```text
Collect the article title, author and publication date. Keep the original
language of the page. If a field is absent, return it in missing.
```

`urls.txt`:

```text
https://habr.com/ru/articles/1103276/
```

Run:

```powershell
web2spec collect `
  --task (Get-Content examples/task.txt -Raw) `
  --urls examples/urls.txt `
  --out data/reports/example_collect.json
```

On Linux/macOS:

```bash
web2spec collect \
  --task "$(cat examples/task.txt)" \
  --urls examples/urls.txt \
  --out data/reports/example_collect.json
```

The output contains `pages`, `aggregate` and `stats`. Facts have exact
`evidence`; absent values are in `missing`, not invented.

## Research a vague question

```powershell
web2spec research `
  --task "Compare the current speech-to-text tariffs of Yandex SpeechKit, Sber SaluteSpeech and VK Cloud: price per hour, free tier, Russian language, diarization. Use official pages and report missing fields." `
  --min-pages 20 --max-pages 30 `
  --out data/reports/example_research.json
```

## MCP client

Use the server config from [`configs/mcp.example.json`](../configs/mcp.example.json).
Then ask the client to call `research` once, or `collect` with a short URL list.

## Offline fixture

For a deterministic test without network, put a page into a directory as
`sha1(url).html` and pass `--html-dir`:

```powershell
web2spec collect `
  --task "Collect the title" `
  --url https://example.com/article/1 `
  --html-dir examples/offline_html `
  --out data/reports/offline.json
```

`examples/offline_html` is not committed with real pages; create it from pages
you are allowed to store.
