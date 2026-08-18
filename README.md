# Personal Website

Source code for [my personal academic website](https://williamwuuu.github.io/), which is built with Jekyll and hosted on GitHub Pages. It began as a fork of [luost26/academic-homepage](https://github.com/luost26/academic-homepage), but has since grown into a substantially customized codebase for publishing my research, notes, and personal writing.

## Highlights

The academic-homepage foundation provides the profile, news, publication, and showcase components. This repository adds a more complete reading and writing experience on top of that foundation:

- **Light, dark, and system themes** with a persistent, keyboard-accessible theme switcher and theme-aware code highlighting.
- **A custom blogs navigation page** with clean post tags, descriptions, and category tabs for **Musings** and **Learnings**.
- **Collapsible proof blocks** with responsive handling for wide mathematical expressions.
- **Elegant code blocks** with syntax highlighting, line numbers, and a copy button.
- **GitHub Discussions comments** powered by [Giscus](https://giscus.app).

## Running Locally

You will need Ruby, Bundler, and the Jekyll prerequisites for your operating system.

```bash
git clone https://github.com/WilliamWuuu/WilliamWuuu.github.io.git
cd WilliamWuuu.github.io
bundle install
bundle exec jekyll serve
```

## Updating the Site

Most routine content changes are data-driven:

- Edit personal details, social links, education, and the CV link in `_data/profile.yml`.
- Edit homepage section visibility in `_data/display.yml`.
- Edit navbar entries in `_data/navigation.yml`.
- Edit shared interface translations in `_data/i18n.yml`.
- Add publication entries under `_publications/<year>/`.
- Add news entries under `_news/`.

### Adding a Blog Post

Create a post at:

```text
_blogs/YYYY-MM-DD-slug/YYYY-MM-DD-slug.md
```

A typical front matter block looks like this:

```yaml
---
layout: blog
title: "Post title"
date: YYYY-MM-DD
description: "A short description used on the blog index."
permalink: /posts/YYYY/M/slug/
image_path: /blog-assets/YYYY-MM-DD-slug/img/
category: notes
tags:
  - Topic
---
```

The two categories currently recognized by the blog index are:

- `original` — displayed as **Musings**
- `notes` — displayed as **Learnings**

Place post images in `blog-assets/YYYY-MM-DD-slug/img/` and insert them with the reusable image widget:

```liquid
{% include widgets/blog_image.html src="figure.png" caption="Figure caption." %}
```

For an optional collapsible proof, use:

```html
<details class="proof" markdown="1">
<summary>Proof</summary>

Write the proof here. Markdown and KaTeX are supported.

</details>
```

Comments are enabled by default for blog posts once Giscus is configured in `_config.yml`. Add `comments: false` to a post's front matter to disable them for that post.

### Bilingual Content

The English site keeps its existing URLs, while Simplified Chinese pages live under `/zh/`. Shared page markup is stored in `_includes/pages/`, so the two languages do not need separate page templates.

- Add short interface copy to both language sections in `_data/i18n.yml`.
- Add `_zh` fields for localized profile, news, publication, and blog-list metadata.
- Pair a fully translated page with `lang` and `translation_url` in its front matter.
- Keep `translation_key` identical across language versions of the same blog post.

If a blog post has no translated counterpart, the Chinese blog index labels its body as English and the article's language switch remains unavailable.

## Deployment

The site is intended for GitHub Pages. Push changes to the publishing branch configured in the repository's Pages settings; GitHub Pages will build and deploy the Jekyll site.

## Acknowledgements

This project was originally based on [luost26/academic-homepage](https://github.com/luost26/academic-homepage). The original project supplied the academic-homepage structure and remains the source of several core components.

## License

The code is available under the [MIT License](LICENSE). The original copyright notice is retained.
