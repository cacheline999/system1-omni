"""Writes workloads.jsonl. Edit here, not the JSONL: the text is fixed so runs stay comparable."""

import json
from pathlib import Path

ROUTE = {
    "type": "choice",
    "instructions": "Route the ticket to the queue that owns it.",
    "criteria": {
        "logistics": "Shipping, delivery and tracking",
        "payment": "Charges, invoices and refunds",
        "returns": "Returns and exchanges",
        "account": "Login, password and profile",
        "human": "Anything else",
    },
}
URGENCY = {
    "type": "score",
    "instructions": "How urgent is the ticket?",
    "criteria": ["Not urgent", "Needs attention soon", "Needs attention immediately"],
}
REFUND = {"type": "noul", "instructions": "Does the customer ask for a refund?"}
ANGRY = {"type": "noul", "instructions": "Is the customer angry?"}
CANCEL = {"type": "noul", "instructions": "Does the customer want to cancel the order?"}
LANG = {
    "type": "choice",
    "instructions": "Which language is the ticket written in?",
    "criteria": {"en": "English", "de": "German", "fr": "French"},
}

SHORT = "My package never arrived and tracking has not updated in ten days."
LONG = (
    "Hello, I ordered a pair of running shoes three weeks ago and paid for express delivery. "
    "The confirmation email said the parcel would arrive within two business days, but the tracking "
    "page has shown 'label created' ever since. I contacted the courier and they told me they never "
    "received the parcel from your warehouse. In the meantime I was charged twice on my credit card, "
    "once for the original amount and once for a slightly different amount that I do not recognise. "
    "I need the shoes for a race next weekend, so please either ship them today with a tracking number "
    "that actually works or cancel the order and refund both charges. I have been a customer for years "
    "and this is the first time something like this has happened."
)
NEAR = " ".join([LONG] * 3)


def options(n):
    names = [
        "logistics",
        "payment",
        "returns",
        "account",
        "human",
        "billing",
        "technical",
        "sales",
        "legal",
        "security",
    ]
    return {
        "type": "choice",
        "instructions": ROUTE["instructions"],
        "criteria": {k: f"The {k} team" for k in names[:n]},
    }


W = [
    # id, kind, state, questions, target tokens per row, measured with check_workloads.py
    ("W1", "bench", SHORT, {"route": ROUTE}, 68),
    ("W2", "bench", LONG, {"route": ROUTE}, 200),
    ("W3", "bench", NEAR, {"route": ROUTE}, 480),
    ("W4", "bench", SHORT, {"route": ROUTE, "urgency": URGENCY, "refund": REFUND}, 54),
    (
        "W5",
        "bench",
        SHORT,
        {"route": ROUTE, "urgency": URGENCY, "refund": REFUND, "angry": ANGRY, "cancel": CANCEL, "lang": LANG},
        49,
    ),
    ("W6", "bench", SHORT, {"refund": REFUND}, 47),
    ("P2", "parity", SHORT, {"route": options(2)}, None),
    ("P5", "parity", SHORT, {"route": options(5)}, None),
    ("P10", "parity", SHORT, {"route": options(10)}, None),
    ("P2L", "parity", LONG, {"route": options(2), "urgency": URGENCY, "refund": REFUND}, None),
    ("P10L", "parity", LONG, {"route": options(10), "urgency": URGENCY, "refund": REFUND}, None),
]

with open(Path(__file__).resolve().parent / "workloads.jsonl", "w") as f:
    f.writelines(
        json.dumps({"id": wid, "kind": kind, "target_tokens_per_row": target, "state": state, "questions": qs}) + "\n"
        for wid, kind, state, qs, target in W
    )
