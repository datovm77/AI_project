import requests
import json
import streamlit as st
import re
import time
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

## python search_service.py
## 联网搜索数据处理数据ai总结数据返回数据模块

# === 配置区域 ===
# 屏蔽列表：跳过这些无法抓取或无关的网站
BLOCKED_SITES = [
    "youtube.com", "youtu.be",
    "twitter.com", "x.com",
    "facebook.com", "instagram.com",
    "linkedin.com", "pinterest.com",
    "tiktok.com", "douyin.com",
    "bilibili.com" # 视频站点通常只有字幕，且容易超时
]

# 请求头：模拟浏览器，避免被反爬
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"
}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["API_KEY"] 
)
MODEL_NAME = "x-ai/grok-4.1-fast"

def ai_extract_json(content, url, max_retries=2):
    """
    将杂乱的网页文本清洗为严格的 JSON 格式 (带重试机制)
    """
    
    # 【保险 2】System Prompt：极其严格的约束
    system_prompt = """
    你是一个不知疲倦的数据提取API。
    任务：阅读用户提供的网页文本，提取信息。
    
    【重要提示】
    1. 网页文本可能包含大量导航菜单、广告或无关链接，请忽略它们，只关注核心正文。
    2. 即使正文被大量导航包裹，只要能找到有价值的内容，就视为有效。

    【严格输出约束】
    1. 你必须只输出 RFC8259 标准的 JSON 字符串。
    2. 不要使用 Markdown 代码块（即不要用 ```json 开头）。
    3. 如果网页内容无效（如全是乱码、验证码、登录页），请将 "valid" 字段设为 false。
    
    【输出 JSON 模版】
    {
        "valid": true,
        "title": "网页标题",
        "summary": "500字以内的总结，提取核心内容，是什么类型提炼出什么类型",
        "key_points": ["关键点1", "关键点2", "关键点3"],
        "code_snippets": ["提取到的关键代码片段(如果有)"],
        "source_url": "原链接"
    }
    """

    # 截断过长内容，防止 Token 溢出或费用过高
    user_prompt = f"原文链接: {url}\n\n原文内容:\n{content[:60000]}" 

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, 
                temperature=0.1, 
                timeout=45 
            )
            
            raw_content = response.choices[0].message.content
            result = clean_and_parse_json(raw_content)
            
            if result:
                # 再次确认 valid 字段
                if not result.get("valid", True):
                    print(f"   [AI判定无效]: {url}")
                    return None
                return result
                
        except Exception as e:
            print(f"   [AI总结重试 {attempt+1}/{max_retries+1}] {e}")
            time.sleep(1) # 避让一下
            
    return None

def clean_and_parse_json(raw_text):
    """
    【保险 3】Python 代码清洗：防止 AI 加了 ```json 导致解析失败
    """
    try:
        # 1. 去掉可能存在的 Markdown 标记
        text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        text = re.sub(r'^```json\s*', '', text, flags=re.MULTILINE) 
        text = re.sub(r'^```\s*', '', text, flags=re.MULTILINE)
        text = text.strip()
        
        # 2. 尝试解析
        return json.loads(text)
    except json.JSONDecodeError:
        print("   [解析失败] JSON 格式错误")
        return None


def _clean_html(raw_html):
    """
    简单清洗 HTML，移除 script/style 和标签，保留纯文本
    """
    # 1. 移除 script 和 style
    text = re.sub(r'<script.*?>.*?</script>', '', raw_html, flags=re.DOTALL)
    text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
    
    # 2. 移除注释
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # 3. 移除 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 4. 处理多余空行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    return text.strip()

def fetch_jina_content(link, max_retries=2):
    """
    尝试使用 Jina Reader 抓取内容。
    如果 Jina 失败 (403/404等)，则回退到直接 requests 抓取 + 简单正则清洗。
    """
    jina_url = f"https://r.jina.ai/{link}"
    
    # --- 阶段 1: 尝试 Jina ---
    for attempt in range(max_retries + 1):
        try:
            read_res = requests.get(jina_url, headers=HEADERS, timeout=(10, 30))
            
            if read_res.status_code == 200:
                content = read_res.text
                # Jina 返回的是 Markdown，做简单清洗
                content_clean = re.sub(r'!\[.*?\]\(.*?\)', '', content)
                content_clean = re.sub(r'\n\s*\n', '\n\n', content_clean)
                return content_clean
                
            elif read_res.status_code in [429, 500, 502, 503, 504]:
                time.sleep(1 * (attempt + 1))
                continue
            else:
                # 403/404 等错误，直接跳出 Jina 重试循环，进入 fallback
                print(f"   [Jina失败 {read_res.status_code}]，尝试直连: {link}")
                break
                
        except Exception as e:
            if attempt < max_retries:
                time.sleep(1)
                continue
            print(f"   [Jina出错]: {str(e)[:50]}")
            
    # --- 阶段 2: Fallback 直连抓取 ---
    print(f"   [Fallback] 直连抓取: {link}")
    try:
        # 直连也尝试 2 次
        for attempt in range(2):
            try:
                # 添加 Referer 可能有助于通过部分反爬
                direct_headers = HEADERS.copy()
                direct_headers["Referer"] = "https://www.google.com/"
                
                res = requests.get(link, headers=direct_headers, timeout=(10, 30))
                

                if res.status_code == 200:
                    # 解决乱码
                    res.encoding = res.apparent_encoding
                    
                    # 只有文本足够长才视为有效
                    if len(res.text) < 500:
                        print(f"   [直连] 内容过短，可能被验证拦截: {link}")
                        continue
                        
                    # 清洗 HTML
                    cleaned_text = _clean_html(res.text)
                    return cleaned_text
                    
                elif res.status_code == 403:
                    print(f"   [直连 403] 依然被拦截: {link}")
                    # 403 通常重试也没用，除非换 IP，这里直接放弃
                    break
                else:
                    time.sleep(1)
            except Exception as e:
                 print(f"   [直连异常]: {e}")
                 time.sleep(1)

    except Exception as e:
        print(f"   [Fallback失败]: {e}")

    return None


def process_single_search_result(idx, item):
    """
    处理单个搜索结果的工作单元 (Thread Worker)
    """
    title = item.get('title')
    link = item.get('link')
    snippet = item.get('snippet')
    
    # 1. 屏蔽检查
    if any(blocked in link for blocked in BLOCKED_SITES):
        print(f"[{idx+1}] 跳过屏蔽网站: {link}")
        return None

    print(f"[{idx+1}] 开始处理: {title[:20]}...")

    # 2. 抓取正文
    content = fetch_jina_content(link)
    
    if not content:
        return None
        
    if len(content) < 300:
        print(f"   [内容无效] 过短({len(content)}字): {link}")
        return None
        
    print(f"   [抓取成功] ({len(content)}字)，正在AI总结...")

    # 3. AI 总结
    structured_data = ai_extract_json(content, link)
    
    if structured_data:
        print(f"   [✅ 处理完成] {title[:15]}...")
        # 补全某些字段防丢失
        if "source_url" not in structured_data:
            structured_data["source_url"] = link
        return structured_data
    else:
        print(f"   [❌ 总结失败] {title[:15]}...")
        return None

def search_for_keyword(query:str):
    url = "https://google.serper.dev/search"
    
    try:
        api_key_search = st.secrets["API_SEARCH"]
    except:
        print(" 没找到 st.secrets，请直接在代码里填入 Key 测试")
        return []
    
    print(f"🚀 正在并发搜索: {query} ...")

    payload = json.dumps({
        "q": query,
        "gl": "cn",
        "hl": "zh-cn",
        "num": 3 
    })
    
    headers = {
        'X-API-KEY': api_key_search,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.request("POST", url, headers=headers, data=payload, timeout=(5, 30))
        search_data = response.json()
    except Exception as e:
        print(f"搜索 API 请求失败: {e}")
        return []

    final_results = []
    
    if "organic" in search_data:
        results_list = search_data["organic"]
        print(f"搜索完成，找到 {len(results_list)} 个原始结果，开启 5 线程并发处理...\n")

        # === 并发处理核心 ===
        with ThreadPoolExecutor(max_workers=5) as executor:
            # 提交任务
            future_to_item = {
                executor.submit(process_single_search_result, idx, item): item 
                for idx, item in enumerate(results_list)
            }
            
            # 获取结果
            for future in as_completed(future_to_item):
                try:
                    data = future.result()
                    if data:
                        final_results.append(data)
                except Exception as exc:
                    print(f"线程执行异常: {exc}")

        print(f"\n🎉 流程结束，有效汇总: {len(final_results)} 篇")
        return final_results
    else:
        print("未找到搜索结果")
        return []
# python search_service.py
if __name__ == "__main__":
    # 测试代码
    res = search_for_keyword("C语言 螺旋矩阵与Z字形遍历 算法")
    print(json.dumps(res, indent=2, ensure_ascii=False))