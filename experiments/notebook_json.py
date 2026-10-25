"""Portable, script-free JSON inspection for saved notebook evidence."""

import html
import json


def readable_json(value, title="Saved request — readable view"):
    """Render decoded strings and expandable containers without changing evidence.

    Unlike a JSON dump, string values contain real line breaks. All values and
    labels are escaped; no scripts, network assets or notebook extensions run.
    """

    def render(label, item, depth):
        label = html.escape(str(label))
        if isinstance(item, (dict, list)):
            children = item.items() if isinstance(item, dict) else enumerate(item)
            kind = "fields" if isinstance(item, dict) else "items"
            opened = " open" if depth < 3 else ""
            body = "".join(render(k, v, depth + 1) for k, v in children)
            return (
                f'<details{opened} style="margin:8px 0 8px 14px">'
                f"<summary><strong>{label}</strong> · {len(item)} {kind}</summary>"
                f"{body}</details>"
            )
        text = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
        return (
            '<div style="margin:10px 0 10px 14px">'
            f"<strong>{label}</strong>"
            '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;'
            "font:inherit;line-height:1.55;padding:10px;border:1px solid #8886;"
            'border-radius:4px;max-height:32em;overflow:auto">'
            f"{html.escape(text)}</pre></div>"
        )

    if value is None:
        return "<p>No request recorded.</p>"
    return render(title, value, 0)
