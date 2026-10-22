---
name: web-scrape
description: Reads and summarizes the content of a specific web page when the user provides a URL (a link starting with http:// or https://) or names a page to read. Uses the scrape_page tool. Do not use for general questions without a link; use web_search for those.
---

# Reading web pages

Use this skill to fetch and understand the content of one specific page. The page is fetched with the `scrape_page` tool.

## When to use

Use this skill when the user:
- Sends a link and asks for a summary, the main points, or specific information from it
- Asks what a page they name says (for example, "what does the pricing page of example.com say?")
- Asks you to compare or extract data from a page they point to

## When NOT to use

- The user asks a general question with no link. Use `web_search` instead.
- The user asks you to find pages or sources. Use `web_search` instead.
- The user asks about a page you have already read in this conversation. Use the earlier result.

## Steps

1. **Get the URL.** It must start with `http://` or `https://`. If the user gave only a domain such as `example.com/about`, add `https://` before calling the tool.
2. **Call `scrape_page`** with the full URL. Call it once per page.
3. **Check the result:**
   - If it contains `error`, go to the error rules below.
   - If `content` is empty or contains only menu items, cookie notices, or navigation text, go to the empty-content rules below.
   - Otherwise, read `title` and `content`.
4. **Check `truncated`.** If it is `true`, the page was longer than the amount returned. Say that your answer covers only the beginning of the page, and ask whether the user wants you to focus on a specific section.
5. **Answer the user.** Write the answer in your own words, using the page's `title` as a reference. End with the URL as the source.

## Rules

- Never invent page content. Only report what the tool returned.
- Do not copy long passages. Summarize, and keep any quoted text under a short phrase.
- Do not read more pages than the user asked for. If the user asks about one page, call the tool once.
- If the user asks for something the page does not contain, say that the page does not mention it. Do not guess.

## Error rules

- If the result says the URL must start with `http://` or `https://`, ask the user for the full link.
- If the result says the page could not be fetched, tell the user the error message in plain language. Suggest that the site may be offline, may block automated access, or may require a login.
- Do not try other URLs on the same site to replace the failed one unless the user asks.

## Empty-content rules

When `content` is empty or has no real text, tell the user:
- The page may load its text with JavaScript, which this tool cannot read.
- The user can copy the text from the page and paste it into the chat, and you can work with that instead.

Do not guess the content from the page title or URL.