/**
 * Theme-aware chart colors.
 *
 * The stylesheet's palette block defines every report color as a
 * `--c-<dark-hex>` variable that [data-theme="light"] remaps. Chart options
 * are plain ECharts/React Flow objects, so they can't use `var()` — `tc`
 * resolves the variable at render time instead. Charts pick up a theme
 * switch because App remounts the whole view tree on toggle.
 */
export type Theme = "dark" | "light";

const STORAGE_KEY = "evalvitals-theme";

/** The color a dark-theme hex literal resolves to under the active theme. */
export function tc(hex: string): string {
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(`--c-${hex.slice(1).toLowerCase()}`)
    .trim();
  return value || hex;
}

/** Saved preference, else whatever the document was served with, else dark.
 * A static export (`static_export.py`, or the pages demo) can pre-set
 * `<html data-theme="light">` to choose its own default. */
export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch { /* storage can be unavailable (file://, privacy mode) */ }
  return document.documentElement.dataset.theme === "light" ? "light" : "dark";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  document.querySelector('meta[name="theme-color"]')
    ?.setAttribute("content", theme === "light" ? "#f7f8fa" : "#07110f");
  try {
    localStorage.setItem(STORAGE_KEY, theme);
  } catch { /* best effort */ }
}
