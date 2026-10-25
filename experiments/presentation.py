"""Small notebook views; all values come from normalized records or saved artifacts."""

import html
import json


def table(rows, columns):
    if not rows:
        return "<p>No saved results for this selection.</p>"

    def value(v):
        if v is None:
            return "Unknown / undefined"
        if isinstance(v, float):
            return f"{v:.8f}".rstrip("0").rstrip(".")
        if isinstance(v, (list, dict)):
            return json.dumps(v)
        return str(v)

    head = "".join(f"<th>{html.escape(k)}</th>" for k in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(value(r.get(k)))}</td>" for k in columns)
        + "</tr>"
        for r in rows
    )
    return f'<div style="overflow-x:auto"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def details(title, content, opened=False):
    text = content if isinstance(content, str) else json.dumps(content, indent=2)
    return f'<details {"open" if opened else ""}><summary>{html.escape(title)}</summary><pre style="white-space:pre-wrap">{html.escape(text)}</pre></details>'


def walkthrough_html(data):
    parts = [
        details("Original attempt record", data["record"], True),
        details("Reproduction evidence", data["reproduction"]),
        details("Agent actions and command outputs", data["actions"]),
        details("Saved patch", data["patch"]),
        details("Original development / acceptance evidence", data["evaluation"]),
        details(
            "Trusted original review (does not change the attempt verdict)",
            data["review"],
        ),
    ]
    if data.get("audit"):
        parts.append(
            details(
                "Later audit clarification — original artifacts preserved",
                data["audit"],
            )
        )
    for diagnostic in data["diagnostics"]:
        parts.append(
            details(
                "Post-run diagnostic evaluation — excluded from success metrics",
                diagnostic,
            )
        )
    return "".join(parts)
