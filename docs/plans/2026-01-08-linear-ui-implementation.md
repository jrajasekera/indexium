# Linear UI Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Indexium's UI from generic DaisyUI defaults to a refined, Linear-inspired professional aesthetic.

**Architecture:** Override DaisyUI CSS variables and add custom component classes in `base.html`. Minimal template changes - primarily CSS work with one structural change to header stats.

**Tech Stack:** Tailwind CSS, DaisyUI (overridden), CSS custom properties

---

## Task 1: Set Up Color System Variables

**Files:**
- Modify: `templates/base.html:13-121` (style block)

**Step 1: Add CSS custom properties for Linear color palette**

In `base.html`, replace the existing `:root` and `[data-theme="dark"]` blocks (lines 14-20) with the new color system:

```css
:root {
    color-scheme: light;
    --bg-base: #f8f9fc;
    --bg-surface: #ffffff;
    --bg-card: rgba(255,255,255,0.8);
    --bg-elevated: rgba(255,255,255,0.9);
    --border-subtle: rgba(0,0,0,0.06);
    --border-hover: rgba(0,0,0,0.1);
    --text-primary: #0f172a;
    --text-secondary: #475569;
    --text-muted: #94a3b8;
    --accent: #8b5cf6;
    --accent-hover: #7c3aed;
    --accent-muted: rgba(139,92,246,0.15);
    --accent-glow: rgba(139,92,246,0.3);
}

[data-theme="dark"] {
    color-scheme: dark;
    --bg-base: #0d0f12;
    --bg-surface: #111318;
    --bg-card: rgba(255,255,255,0.02);
    --bg-elevated: rgba(255,255,255,0.04);
    --border-subtle: rgba(255,255,255,0.06);
    --border-hover: rgba(255,255,255,0.1);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #8b5cf6;
    --accent-hover: #a78bfa;
    --accent-muted: rgba(139,92,246,0.15);
    --accent-glow: rgba(139,92,246,0.3);
}
```

**Step 2: Start the Flask app to verify no CSS errors**

Run: `cd /Users/jrajasekera/source/indexium/.worktrees/linear-ui && uv run python app.py`

Expected: App starts without errors, visit http://localhost:5001

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): add Linear color system CSS variables"
```

---

## Task 2: Override DaisyUI Base Colors

**Files:**
- Modify: `templates/base.html` (style block, after color variables)

**Step 1: Add DaisyUI theme overrides**

Add after the color variables, before the `body` selector:

```css
[data-theme="light"],
[data-theme="dark"] {
    --fallback-b1: var(--bg-base);
    --fallback-b2: var(--bg-surface);
    --fallback-b3: var(--bg-card);
    --fallback-bc: var(--text-primary);
    --fallback-p: var(--accent);
    --fallback-pc: #ffffff;
    --fallback-s: var(--accent);
    --fallback-sc: #ffffff;
    --fallback-a: var(--accent);
    --fallback-ac: #ffffff;
}

[data-theme="dark"] {
    --b1: 13 15 18;
    --b2: 17 19 24;
    --b3: 255 255 255 / 0.02;
    --bc: 241 245 249;
    --p: 139 92 246;
    --pc: 255 255 255;
    --s: 139 92 246;
    --sc: 255 255 255;
    --a: 139 92 246;
    --ac: 255 255 255;
    --n: 30 35 42;
    --nc: 241 245 249;
}

[data-theme="light"] {
    --b1: 248 249 252;
    --b2: 255 255 255;
    --b3: 255 255 255 / 0.8;
    --bc: 15 23 42;
    --p: 139 92 246;
    --pc: 255 255 255;
    --s: 139 92 246;
    --sc: 255 255 255;
    --a: 139 92 246;
    --ac: 255 255 255;
    --n: 241 245 249;
    --nc: 15 23 42;
}
```

**Step 2: Verify theme toggle still works**

Run app, toggle between light/dark mode, verify colors change appropriately.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): override DaisyUI theme colors with Linear palette"
```

---

## Task 3: Restyle Cards

**Files:**
- Modify: `templates/base.html` (style block)

**Step 1: Add card override styles**

Add to the style block:

```css
.card {
    background: var(--bg-card) !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.08) !important;
    transition: border-color 150ms cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-theme="dark"] .card {
    box-shadow: 0 4px 24px rgba(0,0,0,0.24) !important;
}

.card:hover {
    border-color: var(--border-hover) !important;
    transform: none !important;
}

.card-body {
    padding: 20px !important;
}

.card-title {
    font-size: 15px !important;
    font-weight: 500 !important;
}
```

**Step 2: Verify cards render with new styling**

Visit http://localhost:5001/people - cards should have subtle borders, no heavy shadows, glassy appearance in dark mode.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): restyle cards with Linear glassmorphism"
```

---

## Task 4: Restyle Buttons

**Files:**
- Modify: `templates/base.html` (style block)

**Step 1: Add button override styles**

```css
.btn {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.btn-primary {
    background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%) !important;
    border: none !important;
    box-shadow: none !important;
}

.btn-primary:hover {
    filter: brightness(1.1);
    box-shadow: 0 0 20px var(--accent-glow) !important;
}

.btn-primary:active {
    transform: scale(0.98) !important;
}

.btn-outline {
    background: transparent !important;
    border: 1px solid var(--border-hover) !important;
}

.btn-outline:hover {
    background: var(--bg-elevated) !important;
    border-color: var(--border-hover) !important;
}

.btn-ghost {
    background: transparent !important;
    border: none !important;
}

.btn-ghost:hover {
    background: var(--bg-elevated) !important;
}

.btn-error {
    background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
    border: none !important;
}

.btn-warning {
    background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%) !important;
    border: none !important;
}

.btn-secondary {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    border: none !important;
}
```

**Step 2: Verify button states**

Test primary, outline, ghost buttons. Verify hover glows work, active press scales down.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): restyle buttons with Linear gradients and states"
```

---

## Task 5: Restyle Inputs

**Files:**
- Modify: `templates/base.html` (style block)

**Step 1: Add input override styles**

```css
.input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 8px !important;
    height: 40px !important;
    font-size: 14px !important;
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-muted) !important;
    outline: none !important;
}

.input::placeholder {
    color: var(--text-muted) !important;
}

.input-bordered {
    border-color: var(--border-subtle) !important;
}
```

**Step 2: Verify input focus states**

Navigate to tag group page, test input focus shows purple ring.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): restyle inputs with Linear focus states"
```

---

## Task 6: Restyle Badges

**Files:**
- Modify: `templates/base.html` (style block)

**Step 1: Add badge override styles**

```css
.badge {
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 2px 8px !important;
    border-radius: 6px !important;
}

.badge-primary {
    background: var(--accent-muted) !important;
    color: var(--accent-hover) !important;
    border: none !important;
}

.badge-outline {
    background: transparent !important;
    border: 1px solid var(--border-subtle) !important;
    color: var(--text-secondary) !important;
}

.badge-sm {
    font-size: 11px !important;
    padding: 1px 6px !important;
}
```

**Step 2: Verify badges appear refined**

Check people list page for badge styling.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): restyle badges with muted accents"
```

---

## Task 7: Restyle Sidebar

**Files:**
- Modify: `templates/base.html` (style block - replace existing .sidebar and .nav-link styles)

**Step 1: Replace sidebar styles**

Find and replace the existing `.sidebar` block (around lines 47-70) and `.nav-link` block (around lines 26-45):

```css
.nav-link {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 10px 12px;
    border-radius: 8px;
    border-left: 2px solid transparent;
    color: var(--text-secondary);
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
}

.nav-link:hover {
    background: var(--bg-elevated);
    color: var(--text-primary);
}

.nav-link.active {
    background: var(--accent-muted);
    border-left-color: var(--accent);
    color: var(--accent-hover);
}

.sidebar {
    position: fixed;
    inset: 0 auto 0 0;
    width: 260px;
    background: var(--bg-surface);
    border-right: 1px solid var(--border-subtle);
    transform: translateX(0);
    transition: transform 0.3s ease-in-out;
    z-index: 50;
}

.sidebar.collapsed {
    transform: translateX(-100%);
}
```

**Step 2: Remove the sidebar::after pseudo-element**

Delete the `.sidebar::after` block entirely (the gradient line effect).

**Step 3: Verify sidebar looks cleaner**

Sidebar should have flat background, subtle border, no heavy shadow.

**Step 4: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): restyle sidebar with Linear minimal aesthetic"
```

---

## Task 8: Simplify Header Stats

**Files:**
- Modify: `templates/base.html:184-216` (navbar-center section)

**Step 1: Replace the stats bar**

Find the `<div class="navbar-center">` section and replace it entirely:

```html
<div class="navbar-center">
    <div class="flex items-center gap-6 text-sm">
        <span class="text-base-content/60">People <span class="text-base-content font-medium ml-1">{{ stats.named_people_count }}</span></span>
        <span class="text-base-content/60">Groups <span class="text-base-content font-medium ml-1">{{ stats.unnamed_groups_count }}</span></span>
        <span class="text-base-content/60">Pending <span class="text-base-content font-medium ml-1">{{ stats.manual_pending_videos }}</span></span>
    </div>
</div>
```

**Step 2: Verify stats display inline**

Header should now show compact inline stats instead of chunky stat cards.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): simplify header stats to inline display"
```

---

## Task 9: Refine Typography and Spacing

**Files:**
- Modify: `templates/base.html` (style block)

**Step 1: Add typography refinements**

```css
body {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    line-height: 1.5;
    color: var(--text-primary);
    background: var(--bg-base);
}

h1, h2, h3, .text-3xl, .text-2xl, .text-xl {
    font-weight: 600 !important;
    line-height: 1.2 !important;
}

.text-base-content\/70,
.text-base-content\/60 {
    color: var(--text-secondary) !important;
}

.divider {
    opacity: 0.3 !important;
}

.divider::before,
.divider::after {
    background: var(--border-subtle) !important;
}
```

**Step 2: Verify typography feels refined**

Text should feel more cohesive, headings less heavy, dividers more subtle.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): refine typography and spacing"
```

---

## Task 10: Add Alert/Toast Refinements

**Files:**
- Modify: `templates/base.html` (style block)

**Step 1: Add alert overrides**

```css
.alert {
    border-radius: 10px !important;
    border: 1px solid var(--border-subtle) !important;
    background: var(--bg-card) !important;
    backdrop-filter: blur(8px) !important;
}

.alert-success {
    border-color: rgba(34, 197, 94, 0.3) !important;
    background: rgba(34, 197, 94, 0.1) !important;
}

.alert-error {
    border-color: rgba(239, 68, 68, 0.3) !important;
    background: rgba(239, 68, 68, 0.1) !important;
}

.modal-box {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
}
```

**Step 2: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): refine alerts and modals"
```

---

## Task 11: Remove Hover Transforms from Templates

**Files:**
- Modify: `templates/people_list.html:25`

**Step 1: Remove card hover animation**

Find this line:
```html
<div class="card bg-base-100 shadow-lg hover:shadow-xl transition-all duration-300 transform hover:-translate-y-1">
```

Replace with:
```html
<div class="card bg-base-100">
```

**Step 2: Verify cards no longer lift on hover**

People list cards should have subtle border change on hover, not lift animation.

**Step 3: Commit**

```bash
git add templates/people_list.html
git commit -m "feat(ui): remove dated hover transforms from people cards"
```

---

## Task 12: Final Polish - Face Selection States

**Files:**
- Modify: `templates/base.html` (replace existing face-checkbox styles around lines 108-120)

**Step 1: Update face selection styles**

```css
.face-checkbox:checked + .face-thumb {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
    box-shadow: 0 0 16px var(--accent-glow);
}

.face-checkbox:checked + .face-thumb .face-thumb-overlay {
    opacity: 1;
}

.face-checkbox:focus-visible + .face-thumb {
    outline: 2px solid var(--accent);
    outline-offset: 2px;
}

.face-thumb {
    transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
}

.face-thumb:hover {
    border-color: var(--border-hover);
}
```

**Step 2: Verify face selection looks refined**

Selected faces should have purple outline with subtle glow.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): refine face selection states"
```

---

## Task 13: Set Dark Mode as Default

**Files:**
- Modify: `templates/base.html:2` and around line 311

**Step 1: Change default theme**

Line 2, change:
```html
<html lang="en" data-theme="light">
```
to:
```html
<html lang="en" data-theme="dark">
```

Line ~311, change:
```javascript
const savedTheme = localStorage.getItem('theme') || 'light';
```
to:
```javascript
const savedTheme = localStorage.getItem('theme') || 'dark';
```

**Step 2: Verify app starts in dark mode**

Fresh browser (or clear localStorage) should show dark theme.

**Step 3: Commit**

```bash
git add templates/base.html
git commit -m "feat(ui): set dark mode as default theme"
```

---

## Task 14: Run Full Test Suite

**Step 1: Run all tests**

```bash
cd /Users/jrajasekera/source/indexium/.worktrees/linear-ui && uv run pytest -q
```

Expected: All 54 tests pass.

**Step 2: Manual visual verification**

- Visit http://localhost:5001 - verify dark mode, sidebar, header stats
- Visit /people - verify card styling, badges
- Visit a tag group - verify buttons, inputs, face selection
- Toggle to light mode - verify colors invert properly

---

## Task 15: Final Commit and Summary

**Step 1: Ensure all changes committed**

```bash
git status
```

Expected: Clean working directory.

**Step 2: Create summary commit if needed**

If any uncommitted changes remain:
```bash
git add -A
git commit -m "feat(ui): Linear UI redesign complete"
```

---

## Completion Checklist

- [ ] Color system variables added
- [ ] DaisyUI theme overrides in place
- [ ] Cards have glassmorphism effect
- [ ] Buttons have gradient + glow states
- [ ] Inputs have refined focus states
- [ ] Badges are muted and subtle
- [ ] Sidebar has minimal styling (no shadow)
- [ ] Header stats are inline/compact
- [ ] Typography is refined
- [ ] Alerts/modals match theme
- [ ] Hover transforms removed from templates
- [ ] Face selection uses accent color
- [ ] Dark mode is default
- [ ] All tests pass
- [ ] Visual verification complete
