# Concatenated Name Matching for Manual Video Review

## Problem

The manual video review page (`/videos/manual/{id}`) suggests person names by matching known people against the filename. The current matching fails when names appear concatenated without spaces.

**Example failure:**
- Person: `"Timothy Jones"`
- Filename: `"timothyjones - 0gskfllxizojnop6zhv1x_source-ETNdThCX-7RzwThVd.mp4"`
- Current behavior: No match (tokenizes `"timothyjones"` as single token, can't match `["timothy", "jones"]`)

## Solution

Add a concatenated substring check after the exact token match but before fuzzy matching.

**New matching priority:**
1. Exact token match → score 1.0, return immediately
2. **Concatenated substring match → score 0.95, return immediately**
3. Fuzzy SequenceMatcher → best score above threshold (default 0.82)

**How it works:**
- Person name `"timothy jones"` → spaceless: `"timothyjones"`
- Candidate `"timothyjones 0gskfllxizojnop6zhv1x..."` → spaceless: `"timothyjones0gskfllxizojnop6zhv1x..."`
- Check: `"timothyjones" in "timothyjones0gskfllxizojnop6zhv1x..."` → match

## Implementation

**File:** `app.py`

**Function:** `_suggest_manual_person_name()`

**Changes:**
1. Compute `spaceless_name` once per person (outside candidate loop)
2. Compute `spaceless_candidate` once per candidate (outside window loop)
3. After exact match check, add substring check:

```python
if spaceless_name in spaceless_candidate:
    return name, 0.95, source
```

**No new dependencies.** Uses built-in string operations.

**No config changes.** The 0.95 score exceeds the default 0.82 threshold.
