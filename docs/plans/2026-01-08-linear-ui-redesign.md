# Linear-Inspired UI Component Overhaul

## Overview

Redesign Indexium's UI components to achieve a professional, enterprise-grade aesthetic inspired by Linear. The approach overrides DaisyUI's design tokens and adds custom component classes while preserving the existing template structure.

## Design Principles

- **Dark mode first**: Dark theme is the polished primary experience
- **Subtle over flashy**: Refinement through restraint
- **Glassy depth**: Layered surfaces with transparency and blur
- **Consistent rhythm**: 16px base grid, predictable spacing

---

## Color System

### Dark Mode (Primary)

| Role | Value | Usage |
|------|-------|-------|
| Background | `#0d0f12` | Page base |
| Surface | `#111318` | Sidebar, elevated areas |
| Card | `rgba(255,255,255,0.02)` | Cards with backdrop-blur |
| Card elevated | `rgba(255,255,255,0.04)` | Modals, dropdowns |
| Border | `rgba(255,255,255,0.06)` | Subtle dividers |
| Border hover | `rgba(255,255,255,0.1)` | Interactive borders |
| Text primary | `#f1f5f9` | Headings, important text |
| Text secondary | `#94a3b8` | Body text |
| Text muted | `#64748b` | Placeholders, hints |
| Accent | `#8b5cf6` | Primary actions |
| Accent hover | `#a78bfa` | Hover states |
| Accent muted | `rgba(139,92,246,0.15)` | Badges, backgrounds |

### Light Mode

| Role | Value |
|------|-------|
| Background | `#f8f9fc` |
| Surface | `#ffffff` |
| Card | `rgba(255,255,255,0.8)` |
| Border | `rgba(0,0,0,0.06)` |
| Text primary | `#0f172a` |
| Text secondary | `#475569` |
| Text muted | `#94a3b8` |

---

## Components

### Cards

```css
.card {
  background: rgba(255,255,255,0.02);
  backdrop-filter: blur(8px);
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 12px;
  box-shadow: 0 4px 24px rgba(0,0,0,0.12);
}

.card:hover {
  border-color: rgba(255,255,255,0.1);
  /* No transform - just border brightening */
}
```

### Buttons

**Primary:**
```css
.btn-primary {
  background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
  border: none;
  border-radius: 8px;
  padding: 8px 16px;
  font-weight: 500;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
}

.btn-primary:hover {
  filter: brightness(1.1);
  box-shadow: 0 0 20px rgba(139,92,246,0.3);
}

.btn-primary:active {
  transform: scale(0.98);
}
```

**Secondary/Outline:**
```css
.btn-outline {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px;
}

.btn-outline:hover {
  background: rgba(255,255,255,0.05);
}
```

**Ghost:**
```css
.btn-ghost {
  background: transparent;
  border: none;
}

.btn-ghost:hover {
  background: rgba(255,255,255,0.04);
}
```

### Inputs

```css
.input {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  height: 40px;
  padding: 0 12px;
  font-size: 14px;
}

.input:focus {
  border-color: #8b5cf6;
  box-shadow: 0 0 0 2px rgba(139,92,246,0.2);
  outline: none;
}

.input::placeholder {
  color: #475569;
}
```

### Badges

```css
.badge {
  background: rgba(139,92,246,0.15);
  color: #a78bfa;
  font-size: 12px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 6px;
}

.badge-outline {
  background: transparent;
  border: 1px solid rgba(255,255,255,0.1);
  color: #94a3b8;
}
```

### Navigation

**Sidebar:**
```css
.sidebar {
  background: #111318;
  border-right: 1px solid rgba(255,255,255,0.06);
  box-shadow: none; /* Remove heavy shadow */
}
```

**Nav links:**
```css
.nav-link {
  padding: 10px 12px;
  border-radius: 8px;
  border-left: 2px solid transparent;
  transition: all 150ms cubic-bezier(0.4, 0, 0.2, 1);
}

.nav-link:hover {
  background: rgba(255,255,255,0.04);
  transform: none; /* Remove translateX */
}

.nav-link.active {
  background: rgba(139,92,246,0.12);
  border-left-color: #8b5cf6;
  color: #a78bfa;
  box-shadow: none; /* Remove glow */
}
```

### Header Stats

Replace chunky stat cards with compact inline display:

```html
<div class="flex items-center gap-6 text-sm">
  <span class="text-muted">People <span class="text-primary font-medium ml-1">24</span></span>
  <span class="text-muted">Groups <span class="text-primary font-medium ml-1">12</span></span>
  <span class="text-muted">Pending <span class="text-primary font-medium ml-1">3</span></span>
</div>
```

---

## Typography

| Element | Size | Weight | Line-height |
|---------|------|--------|-------------|
| Page title | 24px | 600 | 1.2 |
| Card title | 15px | 500 | 1.3 |
| Body | 14px | 400 | 1.5 |
| Small/muted | 13px | 400 | 1.4 |
| Badge | 12px | 500 | 1 |

---

## Spacing

- Base unit: 16px
- Card padding: 20px
- Section gaps: 24px
- Related element gaps: 12px
- Component internal gaps: 8px

---

## Transitions

- Hovers: 150ms
- State changes: 200ms
- Easing: `cubic-bezier(0.4, 0, 0.2, 1)`

---

## Implementation Approach

1. Override DaisyUI CSS variables in `base.html` `<style>` block
2. Add custom component classes for Linear-specific styling
3. Update templates only where structural changes needed (header stats)
4. Test both dark and light modes

### Files to Modify

- `templates/base.html` - Design tokens, component overrides, header stats
- Individual templates - Minimal changes (remove hover transforms, update classes)

---

## Removed/Changed Patterns

| Current | New |
|---------|-----|
| `shadow-lg` on cards | Subtle border + minimal shadow |
| `hover:-translate-y-1` | Border color change only |
| Colored stat figures | Monochrome icons |
| Heavy sidebar shadow | Single border |
| `translateX` nav hover | Background only |
| Bright DaisyUI colors | Muted purple accent |
