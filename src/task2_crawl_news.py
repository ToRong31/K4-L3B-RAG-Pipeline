"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://tuoitre.vn/ba-diem-moi-trong-ky-tuyen-sinh-nam-hoc-2024-2025-cua-vinuni-20231016164859885.htm",
    "https://dantri.com.vn/giao-duc/cap-hoc-bong-ho-tro-tai-chinh-hao-phong-vinuni-lieu-co-thu-hut-duoc-nhan-tai-20191113144501267.htm",
    "https://vietnamnet.vn/vong-tuyen-sinh-dac-biet-cho-hoc-sinh-tai-nang-vao-dai-hoc-vinuni-718887.html",
    "https://dantri.com.vn/giao-duc/du-an-dai-hoc-vinuni-cong-bo-huong-tuyen-sinh-nam-hoc-2020-2021-20191111111626849.htm",
    "https://dansinh.dantri.com.vn/nhan-luc/vinuni-mo-vong-tuyen-sinh-dac-biet-thu-hut-sinh-vien-quoc-te-va-cac-tai-nang-tam-co-quoc-te-20210311152956321.htm",
]


async def crawl_article(url: str) -> dict:
    from datetime import datetime
    import requests
    from bs4 import BeautifulSoup

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    title_elem = soup.find("h1") or soup.find("title")
    title = title_elem.get_text(strip=True) if title_elem else "Tin tức tuyển sinh VinUni"

    sapo = soup.select_one(".detail-sapo, .singular-sapo, .article-sapo, .content-detail-sapo")
    sapo_text = f"**{sapo.get_text(strip=True)}**\n\n" if sapo else ""

    body = soup.select_one(
        ".detail-content, .singular-content, .article-body, .content-detail, #main-detail-body, .detail__content"
    )
    elems = body.find_all(["p", "h2", "h3"]) if body else soup.find_all(["p", "h2", "h3"])

    paragraphs = []
    for elem in elems:
        text = elem.get_text(strip=True)
        if len(text) > 20 and not any(skip in text.lower() for skip in ["bình luận", "chia sẻ bài viết", "tag:"]):
            if elem.name in ["h2", "h3"]:
                paragraphs.append(f"### {text}")
            else:
                paragraphs.append(text)

    content_markdown = sapo_text + "\n\n".join(paragraphs)

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": content_markdown,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
