/**
 * Smoke test: a real run's report opens on every view without a console error.
 *
 * The payload is what `evalrx report` embeds in a portable export
 * (fixtures/README.md says how it was made). Media is not embedded, so the
 * `/api/media` and `/api/artifact` requests the views make are answered with a
 * placeholder image — the test is about the app, not the files.
 */
import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";

const payload = JSON.parse(readFileSync(new URL("../fixtures/chartqa-gemini.json", import.meta.url), "utf8"));
const PIXEL = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
  "base64",
);

async function openReport(page: Page): Promise<string[]> {
  const errors: string[] = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.route("**/api/**", (route) => route.fulfill({ contentType: "image/png", body: PIXEL }));
  await page.addInitScript((data) => {
    (window as unknown as { __EVALRX_REPORT__: unknown }).__EVALRX_REPORT__ = data;
  }, payload);
  await page.goto("/");
  return errors;
}

const toOverview = (page: Page) =>
  page.getByRole("button", { name: "Overview" }).click();
const openAuditView = (page: Page, view: string) =>
  page.locator(".evidence-index").getByRole("button", { name: view }).click();

test("every view of a real run renders without console errors", async ({ page }) => {
  const errors = await openReport(page);
  const stages: Array<{ code: string; title: string }> = payload.data.stages;

  await expect(page.locator(".topbar")).toContainText(payload.data.trace_id.slice(0, 8));

  await openAuditView(page, "evidence");
  for (const stage of stages) {
    await page.locator(".stage-list").getByRole("button", { name: new RegExp(stage.title) }).click();
    await expect(page.locator(".evidence-head h2")).toHaveText(stage.title);
    await expect(page.locator(".detail-page nav")).toContainText("What this step found");
    await page.getByRole("tab", { name: "Full record" }).click();
    await expect(page.locator(".detail-page nav")).toContainText("Everything this step recorded");
  }

  await toOverview(page);
  await openAuditView(page, "cases");
  await expect(page.locator(".case-detail h2")).toHaveText(payload.data.cases[0].id);
  await page.locator(".case-toolbar").getByRole("button", { name: "fail", exact: true }).click();
  await expect(page.locator(".case-detail")).toBeVisible();

  await toOverview(page);
  await openAuditView(page, "debug");
  await expect(page.locator(".detail-page")).toContainText(payload.data.debug.events[0].event);

  expect(errors).toEqual([]);
});
