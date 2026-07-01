# TLA DSL Docs

This documentation set is a lightweight MkDocs site for the TLA DSL.

The API reference is generated into Markdown first, then MkDocs builds it into
static HTML with a left-side navigation bar and page table of contents. Treat
the generated Markdown as a build source, not as the primary reading format.

## Local Preview

From `python/tla_dsl`:

```bash
python3 -m mkdocs serve
```

To build static HTML:

```bash
python3 -m mkdocs build
```

Then open `site/index.html` in a browser.
