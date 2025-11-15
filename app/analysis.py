"""
Anomaly Detection Script for Aurora Messages Dataset
Run:
    python -m app.analysis
"""

import os
import re
import json
from collections import defaultdict, Counter
from datetime import datetime, timedelta

from dateutil.parser import parse as date_parse
from openai import OpenAI

from .qa import fetch_messages


# ---------- Helper: burst detection ----------

def detect_burst_messages(messages, threshold_count=5, time_window_seconds=30):
    """
    Detect burst messaging patterns:
      - Same user sends >= threshold_count messages
      - Within a given time window (in seconds)
    """
    bursts = []
    by_user = defaultdict(list)

    # Group messages by user
    for m in messages:
        user = m.get("member_name", "Unknown")
        by_user[user].append(m)

    for user, msgs in by_user.items():
        # Sort by timestamp
        valid_msgs = [m for m in msgs if m.get("timestamp")]
        if not valid_msgs:
            continue

        valid_msgs.sort(key=lambda x: x["timestamp"])
        times = [date_parse(m["timestamp"]) for m in valid_msgs]

        # Sliding window over timestamps
        for i in range(len(times) - threshold_count + 1):
            start = times[i]
            end = times[i + threshold_count - 1]
            if (end - start).total_seconds() <= time_window_seconds:
                bursts.append(
                    {
                        "user": user,
                        "count": threshold_count,
                        "start": valid_msgs[i]["timestamp"],
                        "end": valid_msgs[i + threshold_count - 1]["timestamp"],
                    }
                )
                break  # flag user once

    return bursts


# ---------- GPT-based underspecified intent detection ----------

def detect_underspecified_requests_gpt(messages, max_examples=80):
    """
    Use GPT to identify underspecified intent requests.

    We first filter to messages that look like "action requests"
    (book / reserve / schedule / arrange / order / buy / etc.),
    then let GPT decide which ones lack enough detail to act on.

    Example of underspecified (should be flagged):
      "Please book a private jet to Paris for this Friday."
       - No passenger count
       - No departure city
       - No exact time

    Example of well-specified (should NOT be flagged):
      "Book me a table for 2 at Le Bernardin on the 15th at 7 PM."
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set; skipping GPT-based underspecified detection.")
        return []

    client = OpenAI(api_key=api_key)

    action_words = ["book", "reserve", "schedule", "arrange", "order", "buy"]
    candidate_msgs = []

    # Filter messages that look like booking / action requests
    for m in messages:
        text = m.get("text", "")
        lower = text.lower()
        if any(a in lower for a in action_words):
            candidate_msgs.append(m)

    if not candidate_msgs:
        return []

    # Limit to avoid huge prompts
    candidate_msgs = candidate_msgs[:max_examples]

    # Build a numbered list for GPT
    lines = []
    for idx, m in enumerate(candidate_msgs, start=1):
        user = m.get("member_name", "Unknown")
        text = m.get("text", "").replace("\n", " ")
        lines.append(f"{idx}. {user}: {text}")

    joined = "\n".join(lines)

    system_prompt = (
        "You are an expert concierge operations assistant.\n"
        "You will be given a list of messages where clients are asking to book, reserve, schedule, "
        "or arrange something.\n\n"
        "Your job is to mark which messages are UNDERSPECIFIED, meaning:\n"
        " - A human concierge could not safely complete the request as written, because key details are missing.\n"
        "Examples of missing details include: date, time, number of people, origin/destination when relevant, "
        "or other critical parameters.\n"
        "If a request has enough information to proceed or at least schedule a follow-up, consider it specified.\n\n"
        "Return ONLY a JSON array of integers representing the line numbers of underspecified messages.\n"
        "If none are underspecified, return an empty JSON array: []."
    )

    user_prompt = (
        "Here are the messages:\n\n"
        f"{joined}\n\n"
        "Which line numbers are underspecified? Answer ONLY with a JSON array of integers, e.g. [1, 4, 7]."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()

        # Try to parse the JSON array
        underspecified_indices = json.loads(content)
        if not isinstance(underspecified_indices, list):
            raise ValueError("GPT did not return a JSON list")

        # Map indices back to messages
        underspecified_msgs = []
        for i in underspecified_indices:
            if isinstance(i, int) and 1 <= i <= len(candidate_msgs):
                underspecified_msgs.append(candidate_msgs[i - 1])

        return underspecified_msgs

    except Exception as e:
        print(f"GPT underspecified detection failed: {e}")
        return []


# ---------- Main analysis ----------

def main():
    print("Fetching messages...\n")
    messages = fetch_messages(limit=500)
    print(f"Loaded {len(messages)} messages\n")

    # Basic stats
    user_names = [m.get("member_name") for m in messages if m.get("member_name")]
    unique_users = set(user_names)
    lengths = [len(m.get("text", "")) for m in messages]
    avg_len = sum(lengths) / len(lengths) if lengths else 0.0

    print("=== DATA STATISTICS ===")
    print(f"Total messages: {len(messages)}")
    print(f"Unique users: {len(unique_users)}")
    print(f"Average message length: {avg_len:.2f} characters\n")

    # ---- Missing required fields ----
    missing_user = [m for m in messages if not m.get("member_name")]
    missing_text = [m for m in messages if not m.get("text")]
    missing_timestamp = [m for m in messages if not m.get("timestamp")]

    # ---- Duplicate message texts ----
    text_map = defaultdict(list)
    for m in messages:
        txt = (m.get("text") or "").strip()
        if txt:
            text_map[txt].append(m["id"])

    duplicate_msgs = {txt: ids for txt, ids in text_map.items() if len(ids) > 1}

    # ---- Very short / empty messages ----
    short_msgs = [m for m in messages if len((m.get("text") or "").strip()) < 5]

    # ---- Impossible / future timestamps ----
    future_cutoff_year = datetime.now().year + 2
    impossible_times = []
    for m in messages:
        ts = m.get("timestamp")
        if not ts:
            continue
        try:
            dt = date_parse(ts)
            if dt.year > future_cutoff_year:
                impossible_times.append(m)
        except Exception:
            # Malformed dates also interesting, but here we just skip
            continue

    # ---- Burst messages by same user ----
    burst_cases = detect_burst_messages(messages)

    # ---- Underspecified intent requests (GPT-based) ----
    underspecified_cases = detect_underspecified_requests_gpt(messages)

    # ---- Print summary ----
    print("=== ANOMALY REPORT ===")
    print(f"- Duplicate messages: {len(duplicate_msgs)}")
    print(f"- Missing user_name: {len(missing_user)}")
    print(f"- Missing message text: {len(missing_text)}")
    print(f"- Missing timestamp: {len(missing_timestamp)}")
    print(f"- Very short / empty messages (<5 chars): {len(short_msgs)}")
    print(f"- Impossible/future timestamps: {len(impossible_times)}")
    print(f"- Burst messaging cases: {len(burst_cases)}")
    print(f"- Underspecified intent requests (GPT-based): {len(underspecified_cases)}")

    print("\n--- SAMPLE DETAILS ---")
    if duplicate_msgs:
        first_txt, first_ids = next(iter(duplicate_msgs.items()))
        print("\nExample duplicate message:")
        print(f"  Text: {first_txt}")
        print(f"  IDs: {first_ids}")

    if burst_cases:
        print("\nExample burst case:")
        print(f"  {burst_cases[0]}")

    if underspecified_cases:
        print("\nExample underspecified request:")
        for i in range(0, len(underspecified_cases)):
            print(underspecified_cases[i].get("text", ""))
        print(f"  {underspecified_cases[0].get('text', '')}")

    print("\n====================================\n")


if __name__ == "__main__":
    main()
