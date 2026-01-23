import requests
import json
import streamlit as st
import re
from openai import OpenAI
## python search_service.py
## 联网搜索数据处理数据ai总结数据返回数据模块
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets["API_KEY"] 
)
MODEL_NAME = "openai/gpt-oss-120b"

def ai_extract_json(content, url):
    """
    将杂乱的网页文本清洗为严格的 JSON 格式
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

    user_prompt = f"原文链接: {url}\n\n原文内容:\n{content[:80000]}" 

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],

            response_format={"type": "json_object"}, 
            temperature=0.1, # 温度越低，格式越稳定
        )
        
        raw_content = response.choices[0].message.content
        return clean_and_parse_json(raw_content)

    except Exception as e:
        print(f"AI 总结出错: {e}")
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
        print("JSON 解析失败，AI 返回了非标准格式。")
        return None
    
def search_for_keyword(query:str):
    url = "https://google.serper.dev/search"
    
    try:
        api_key_search = st.secrets["API_SEARCH"]
    except:
        print(" 没找到 st.secrets，请直接在代码里填入 Key 测试")
        return
    
    print("正在 Google 搜索 ...")

    payload = json.dumps({
        "q": query,  # 搜索关键词
        "gl": "cn",                 # 地区: 中国 (cn), 美国 (us)
        "hl": "zh-cn",              # 语言: 简体中文
        "num": 5                   # 返回结果数量
    })
    
    headers = {
        'X-API-KEY': api_key_search,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload,timeout=(5,30))
    search_data = response.json()
    result = []
    BLOCKED_SITES = []
    if "organic" in  search_data:
        results_list = search_data["organic"]
        print(f"搜索完成，一共找到{len(results_list)}个结果，开始阅读....\n")

        for idx,item in enumerate(results_list):
            title = item.get('title')
            link = item.get('link')
            snippet = item.get('snippet')

            print("*"*60)
            print(f" 第 {idx+1} 篇: {title}")
            print(f" 链接: {link}")
            if any(blocked in link for blocked in BLOCKED_SITES):
                print(f"跳过屏蔽网站：{link}")
                print("-" * 30)
                continue 
            print(f" 摘要: {snippet}")
            print("-" * 30)

            if link:
                print(f" 正在抓取正文 (Jina Reader)...")
                jina_url = f"https://r.jina.ai/{link}"

                try:
                    read_res = requests.get(jina_url,timeout = 10)

                    if read_res.status_code == 200:
                        content = read_res.text
                        content_clean = re.sub(r'!\[.*?\]\(.*?\)', '', content)
                        content_clean = re.sub(r'\n\s*\n', '\n\n', content_clean)
                        if len(content_clean) < 300:
                            print(f"内容过短 ({len(content_clean)} 字符)，判定为无效页面,跳过。")
                        else:
                            print(f"内容有效 ({len(content_clean)}字)，正在调用 AI 进行结构化总结....")
                            
                            # === 调用刚才写的 AI 函数 ===
                            structured_data = ai_extract_json(content_clean, link)
                            
                            if structured_data:
                                result.append(structured_data)
                    else:
                        print(f" 抓取失败 (状态码 {read_res.status_code})")
                except Exception as e:
                    print(f"抓取超时或出错: {e}")

            print("\n")

        return result
    else:
        print("未找到搜索结果")

if __name__ == "__main__":
    search_for_keyword("插入排序代码")