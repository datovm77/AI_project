import streamlit as st
import asyncio
import os
import json
import base64
import pdfplumber
import docx
from datetime import datetime
from typing import List, Dict, Any, Tuple, AsyncGenerator
from dotenv import load_dotenv
from openai import AsyncOpenAI
try:
    import search_service
except ImportError:
    st.error("找不到 search_service.py，请确保该文件在同一目录下。")



# streamlit run agent1.1.py
# ==========================================
# 1. ⚙️ 配置与初始化
# ==========================================
load_dotenv()  #导入secrets
# nest_asyncio.apply()  # 允许嵌套事件循环

PROFILE_PATH = "profile.txt"
HISTORY_PATH = "history.json"

API_KEY = st.secrets["API_KEY"]
BASE_URL = "https://openrouter.ai/api/v1"

# 初始化客户端
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# 【重要配置】模型角色分配
# 这里的模型选择决定了是否支持多模态 (Vision)
MODEL_CONFIG = {
    "librarian": "google/gemini-3-flash-preview", 
    "reviewer": "google/gemini-3-flash-preview",
    "architect": "google/gemini-3-flash-preview",
    "mentor": "anthropic/claude-opus-4.5"          
}

# ==========================================
# 2. 🛠️ 核心工具函数 (Utils)
# ==========================================

def encode_image_to_base64(image_bytes: bytes) -> str:
    """
    [工具] 将图片二进制流转换为 Base64 字符串。
    用于将图片传给支持 Vision 的 LLM。
    """
    return base64.b64encode(image_bytes).decode('utf-8')

def parse_uploaded_file(uploaded_file) -> Dict[str, Any]:
    """
    [核心工具] 通用文件解析工厂。
    输入: Streamlit 上传文件对象
    输出: 字典 {'filename':..., 'type': 'code'/'document'/'image'/'error', 'content':...}
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    result = {
        "filename": uploaded_file.name,
        "type": "unknown",
        "content": ""
    }

    try:
        if file_type == 'pdf':
            with pdfplumber.open(uploaded_file) as pdf:
                text_parts = []
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:  # ✅ 检查是否为 None
                        text_parts.append(page_text)
                text = '\n'.join(text_parts)
            result["type"] = "document"
            result["content"] = text if text else "[PDF 无法提取文本，可能是扫描版]"

        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            text = "\n".join(para.text for para in doc.paragraphs)
            result["type"] = "document"
            result["content"] = text

        elif file_type in ['txt', 'c', 'cpp', 'py', 'java', 'md', 'js', 'ts', 'go', 'rs']:
            text = uploaded_file.read().decode("utf-8", errors='ignore')
            result["type"] = "code"
            result["content"] = text

        elif file_type in ['png', 'jpg', 'jpeg', 'gif', 'webp']:
            bytes_data = uploaded_file.getvalue()
            result["type"] = "image"
            result["content"] = encode_image_to_base64(bytes_data)

    except Exception as e:
        result["type"] = "error"
        result["content"] = f"读取文件 {uploaded_file.name} 时出错: {str(e)}"

    return result



async def search_web_tool(query: str) -> str:
    """
    [AI 接口] 统一封装的 LLM 调用函数 (Generator)。
    """
    print(f"🔍 [Agent] 正在调用搜索工具，关键词: {query} ...")
    
    try:
        raw_results = await asyncio.to_thread(search_service.search_for_keyword, query)
        
        if not raw_results:
            return f"【搜索结果】关于 '{query}' 未找到有效的网络信息。"

        # 将 JSON 对象拼接成清晰的文本报告供 LLM 阅读
        formatted_report = f"以下是关于 '{query}' 的联网搜索结果汇总：\n\n"
        
        for idx, item in enumerate(raw_results):
            # 容错获取字段，防止某些字段缺失
            title = item.get('title', '未知标题')
            url = item.get('source_url', '#')
            summary = item.get('summary', '暂无摘要')
            key_points = item.get('key_points', [])
            code_snippets = item.get('code_snippets', [])

            formatted_report += f"--- 来源 [{idx + 1}] : {title} ---\n"
            formatted_report += f"链接: {url}\n"
            formatted_report += f"摘要: {summary}\n"
            
            if key_points:
                formatted_report += "关键点:\n"
                for point in key_points:
                    formatted_report += f"   - {point}\n"
            
            if code_snippets:
                formatted_report += "相关代码片段:\n"
                for code in code_snippets:
                    formatted_report += f"```\n{code[:1500]}...\n```\n"
            
            formatted_report += "\n"

        return formatted_report

    except Exception as e:
        error_msg = f"搜索工具调用失败: {str(e)}"
        print(error_msg)
        return error_msg

async def call_ai_chat(model: str, system_prompt: str, user_content: str, image_base64_list: List[str] = None):
    """
    [AI 接口] 统一封装的 LLM 调用函数。
    """
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    if not image_base64_list:
        messages.append({"role": "user", "content": user_content})
    else:
        content_payload = []
        if user_content:
            content_payload.append({"type": "text", "text": user_content})
            
        # 添加图片
        for img_b64 in image_base64_list:
            content_payload.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img_b64}" 
                }
            })
            
        messages.append({"role": "user", "content": content_payload})

    try:
        # 3. 发起异步调用
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            stream=True 
        )
        async for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    except Exception as e:
        yield f"[Ai调用失败]:{str(e)}"
# ==========================================
# 3. Agent 核心逻辑 (Agents)
# ==========================================

# --- Phase 1: 预处理 ---

async def agent_librarian(uploaded_files) -> Tuple[Dict[str, Any], str]:
    """
    [Librarian - 档案管理员]
    职责：清洗数据，分类整理，不进行深度分析。
    """
    # TODO:
    # 1. 遍历 uploaded_files
    # 2. 调用 parse_uploaded_file 解析每个文件
    # 3. 将结果分类放入 list: codes[], docs[], images[]
    # 4. 返回结构化字典 structured_context

    #1.context 字典嵌列表
    context = {"code":[],"docs":[],"images":[]}
    for file in uploaded_files:
        parsed_data = parse_uploaded_file(file)
        if parsed_data['type'] == 'code':
            context['code'].append(parsed_data['content'])
        elif parsed_data['type'] == 'image':
            context["images"].append(parsed_data['content'])
        elif parsed_data["type"] == 'document':
            context["docs"].append(parsed_data['content'])

    current_profile = ""
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH,"r",encoding="utf-8") as f:
            current_profile = f.read()
    else:
        current_profile = "这是用户第一周，暂无个人能力档案。"

    return context,current_profile

async def agent_librarian_write(code_list: List[str]) -> str:
    """
    [Librarian - 档案管理员 (写操作)]
    职责：直接读取本地旧档案，并结合本周上传的【原始代码】，更新 profile.txt。
    参数：code_list (包含本周所有代码文本的列表)
    """   

    # 读取旧档案
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            old_profile = f.read()
    else:
        old_profile = "【新用户】暂无历史档案。初始评级：未定。"

    #预处理代码内容
    raw_code_content = "\n\n--- Next File ---\n\n".join(code_list)
    # if len(raw_code_content) > 50000: 
    #     raw_code_content = raw_code_content[:50000] + "\n...(代码过长已截断)..."

    system_prompt = """
    【任务指令】
    执行技术档案的增量更新任务。基于输入的[旧档案]与[本周原始代码]，输出一份**证据导向**、层级分明的最新技术档案。

    【核心原则】
    1. **拒绝空洞 (No Vague Terms)**：**严禁**使用“掌握”、“熟练”、“了解”等主观形容词。必须用“**动作+结果**”的形式描述。
    2. **证据强制 (Evidence-Based)**：每一项技术能力**必须**附带简短的例子（代码中的具体应用场景、函数名或解决的问题）作为佐证。
    3. **增量保留**：保留[旧档案]所有条目。仅在有新证据时追加内容，**严禁删除**历史记录。

    【输出格式规范】
    档案必须按以下 Markdown 层级结构组织：

    # I. [技术栈大类](例如：Python、前端技术、DevOps、C/C++算法、Java)
    ## [序号].[子序号] [具体领域/库]
    - [动词+技术点]：[简短的代码证据/应用场景]
    - [动词+技术点]：[简短的代码证据/应用场景]

    *❌ 错误写法：*
    - 熟练使用 Python 异步编程
    - 掌握 Streamlit

    *✅ 正确写法示例：*
    # I. Python
    ## 1.1 并发与异步 IO
    - **实现并发任务调度**：在 `agent_reviewer` 模块中使用 `asyncio.gather` 并行执行搜索任务，提升响应速度。
    - **优化内存占用**：利用 `yield` 生成器构建流式数据处理管道，避免一次性加载大文件。

    # II. Web 全栈
    ## 2.1 Streamlit 框架
    - **构建状态管理机制**：使用 `st.session_state` 跨页面/跨刷新持久化存储用户对话历史。
    - **开发自定义组件**：封装 `parse_uploaded_file` 函数工厂，统一处理 PDF/Docx/Code 多格式文件解析。

    # III. C/C++算法
    ## 3.1 .....

    【执行步骤】
    1. **证据提取**：扫描代码，识别技术点，并立刻找到它在代码中“具体解决了什么问题”或“具体实现在哪里”。
    2. **动作化描述**：将“使用了 X 库”转化为“利用 X 库实现了 Y 功能”。
    3. **增量写入**：将新发现的能力条目追加到对应分类下，输出完整的档案。
    """
    
    user_content = f"【当前旧档案】:\n{old_profile}\n\n【本周原始代码堆 (Raw Code Data)】:\n{raw_code_content}"

    model = MODEL_CONFIG["librarian"]

    new_profile_content = ""
    
    #调用 AI 生成新档案
    try:
        async for chunk in call_ai_chat(model, system_prompt, user_content):
            new_profile_content += chunk
    except Exception as e:
        return f"[档案更新失败]: {str(e)}"
    
    #写入文件 (覆盖更新)
    try:
        with open(PROFILE_PATH, "w", encoding="utf-8") as f:
            f.write(new_profile_content)
        print("[Librarian] profile.txt 已根据原始代码更新完毕。")
    except Exception as e:
        return f"[文件写入错误]: {str(e)}"

    return new_profile_content

async def agent_reviewer(context: Dict) -> AsyncGenerator[str, None]:
    """
    [Reviewer - 代码审计员]
    架构升级：Planner (生成搜索词) -> Executor (并行搜索) -> Generator (流式产出)
    """
    code_snippets = context.get('code',[])
    image_list = context.get('images',[])
    full_code_text = "\n\n".join(code_snippets)
    if len(full_code_text) > 30000:
        full_code_text = full_code_text[:30000] + "\n\n(代码过长，后续部分已截断...)"
    if not full_code_text and not image_list:
        yield "[审计员]：未检测到有效代码或截图，无法执行审计。"
        return
    yield "🤔 **[AI 思考中]** 正在分析代码技术栈，规划搜索路径...\n\n"

    planner_prompt = """
    你是一个技术审计规划师。请分析用户的代码，提取出总共 3 个最重要的的最新的技术关键词或知识点，用于后续的联网搜索以获取相关资料。
    
    【搜索目的】
    1. 查找代码所用框架（如 Streamlit, LangChain, PyTorch 等）的最新**官方文档**。
    2. 查找针对当前代码逻辑的**最佳学习内容**或**最新标准写法**。
    3. 查找与代码难度或者知识点匹配的**练习题**（LeetCode/Kaggle/GitHub）。

    【输出格式】
    必须且仅输出一个 Python 列表格式的字符串(加上明显的后缀，如"题目" "官方文档")，例如：
    ["Streamlit 官方文档", "Python asyncio 题目", "RAG system GitHub项目"]
    """
    planner_model = MODEL_CONFIG["reviewer"]
    search_queries = []

    try:
        # 这里我们不流式，直接拿到完整结果
        planner_response = ""
        async for chunk in call_ai_chat(planner_model, planner_prompt, f"【代码内容】:\n{full_code_text[:10000]}"):
            planner_response += chunk
        
        # 清洗并解析 JSON
        clean_json = planner_response.replace("```json", "").replace("```", "").strip()
        search_queries = json.loads(clean_json)
        
        # 容错：如果 AI 返回的不是列表，强制转为列表
        if not isinstance(search_queries, list):
            search_queries = [str(search_queries)]
            
    except Exception as e:
        # 降级策略：如果规划失败，使用默认词
        print(f"[Planner Error]: {e}")
        search_queries = ["本周最佳GitHub开源项目"]


    #开始调用联网搜索工具
    search_results_context = ""
    if search_queries:
        # 实时反馈给用户正在搜什么
        yield f"🌐 **[联网检索]** 正在并行搜索权威资料：\n"
        for q in search_queries:
            yield f"- *检索：{q}*\n"
        yield "\n"

        # 并行执行搜索任务 (使用 asyncio.gather 提速)
        try:
            tasks = [search_web_tool(query) for query in search_queries]
            results = await asyncio.gather(*tasks)
            search_results_context = "\n\n".join(results)
        except Exception as e:
            search_results_context = f"搜索过程发生错误: {str(e)}"

    yield "📝 **[生成报告]** 资料检索完毕，正在撰写深度审计报告...\n\n---\n\n"

    system_prompt = """
    【任务定义】
    依据提供的[代码片段]、[运行截图]及前序步骤获取的[联网参考资料]，撰写严格的代码审计报告。

    【输入说明】
    1. **待审计代码**：用户的原始代码。
    2. **联网参考资料**：系统已提前检索到的官方文档、最佳实践或练习题数据，这不是用户的数据，这是联网所得数据。

    【执行流程】
    1. **视觉诊断 **：若包含图片（报错/运行截图），优先解析错误信息，并定位代码中的具体致错行。
    2. **安全扫描**：检测关键漏洞（SQL注入、XSS、硬编码密钥、敏感数据泄露、越权访问）。
    3. **健壮性评估**：识别运行时风险（空指针、未捕获异常、死循环、资源未关闭、语法错误）。
    4. **代码异味**：指出不可读命名、魔法数字、冗余逻辑或反模式写法。
    2. **资料整合**：
    - **验证**：利用[联网参考资料]校验代码中的API用法或者其它较新甚至陌生的写法是否过时或错误。
    - **推荐**：从[联网参考资料]中提取适合当前代码水平的**练习题链接**或**官方文档链接**。

    【输出板块】(Markdown)
    仅包含以下板块（无内容则省略）：
    - **🔴 致命问题**：导致崩溃或严重安全隐患的错误（罗列错误，展示相应代码片段，可以展示修复后的代码）。
    - **🟡 改进建议**：逻辑简化与代码规范。
    - **📸 截图分析**：针对报错截图的技术解读。
    - **💡 修复代码**：针对严重问题的最小化修复方案。
    - **📚 扩展与参考**：**强制**在此处罗列[联网参考资料]中提供的核心链接（如官方文档URL、练习题URL），然后再补充其中遗漏的知识点文档链接。

    【风格约束】
    客观、直接。严禁忽略提供的[联网参考资料]，严禁输出寒暄语。
    """

    user_content_for_review = f"""
    【待审计代码】:
    {full_code_text}

    【联网参考资料 (非审计内容，为参考内容)】:
    {search_results_context}
    """
    model = MODEL_CONFIG["reviewer"] 
    try:
        async for chunk in call_ai_chat(model, system_prompt, user_content_for_review, image_base64_list=image_list):
            yield chunk

    except Exception as e:
        error_msg = f"\n\n[reviewer 运行出错]: {str(e)}"        
        print(error_msg)
        yield error_msg


async def agent_architect(context: Dict) -> AsyncGenerator[str, None]:
    """
    [Architect - 技术架构师]
    职责：性能评估、技术栈对比、成长值计算。
    """
    #读取旧档案 
    old_profile = ""
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                old_profile = f.read()
        except Exception:
            old_profile = "读取档案出错，视为空白档案。"
    else:
        old_profile = "【新用户】暂无历史档案。初始评级：未定。"

    #提取并清洗数据
    code_snippets = context.get('code', [])
    word_snippets = context.get('docs', [])


    full_code_text = "\n\n".join(code_snippets)
    if len(full_code_text) > 30000:
        full_code_text = full_code_text[:30000] + "\n\n(代码过长，后续部分已截断...)"
    
    full_doc_text = "\n\n".join(word_snippets)
    if len(full_doc_text) > 30000:
        full_doc_text = full_doc_text[:30000] + "\n...(文档过长已截断)"

    #空内容检查
    if not full_code_text and not full_doc_text:
        yield "【架构师】: 未检测到代码或技术文档，无法进行架构评估。"
        return
    
    system_prompt = """
    【指令目标】
    基于用户[旧档案]与[本周代码/文档]，执行宏观架构/代码设计性能评估与技术成长判定。忽略具体语法错误，专注代码的可维护性、设计逻辑、技术上限与运行效率（重点，看架构，看时间复杂度）。

    【执行步骤】
    1. **架构分析**：提取代码的结构模式（如分层、模块化程度）。识别是否应用了特定设计模式（OOP、FP、单例、工厂等），是否优化了运行效率，评估架构性能。
    2. **成长比对**：将本周代码的技术深度与[旧档案]进行对比。
    - 判定状态：**突破**（应用了新概念/新技术/新算法）、**巩固**（熟练度提升）或 **停滞**。
    3. **技术栈提取**：罗列代码中使用的核心框架、第三方库或核心算法。
    4. **综合定级**：根据代码的工程复杂度与设计美感或者运行效率，给出 S/A/B/C 评级。
    5. **不足评估**：根据代码的工程架构找出性能与架构上的不足点，如可维护性、设计逻辑、技术上限与运行效率的不足点。
    【输出格式】
    严格遵循 Markdown 格式，仅输出以下四个板块：

    - **🏗️ 架构/代码运行效率**：(罗列代码结构，运行效率（练习算法就看时间复杂度，是项目就看架构的效率）及模块划分（如果是项目）)
    - **📈 成长评估**：(明确指出与旧档案相比的进步点，重点指出不足点，列出效率低下内容并且给出优化案例。此部分为主要部分，输出贴合最大输出上限)
    - **🛠️ 技术栈侦测**：(列出检测到的关键技术/库/算法)
    - **⚖️ 综合评级**：(给出 S/A/B/C 评分并简述理由)
    - **🛠️ 扩展参考**：(了解架构/算法不成熟的地方，推荐官方文档阅读或者开源项目或者与知识点相关的题目)
    """
    user_content = f"【当前旧档案】:\n{old_profile}\n\n【本周原始代码堆】:\n{full_code_text} \n\n【本周文档内容】:\n{full_doc_text}"

    model = MODEL_CONFIG["architect"]

    try:
        async for chunk in call_ai_chat(model, system_prompt, user_content):
            yield chunk
    except Exception as e:
        error_msg = f"\n\n[Architect 运行出错]: {str(e)}"        
        print(error_msg)
        yield error_msg

async def agent_mentor(review_res: str, architect_res: str, user_note: str,context:Dict) -> AsyncGenerator[str, None]:
    """
    [Mentor - 导师]
    职责：汇总报告，生成最终周报。
    """
    code_snippets = context.get('code', [])
    system_prompt = """
    【指令目标】
    基于[代码审计报告]、[架构评估报告]、[学生心得]及[学生源代码]，撰写一份综合性的《本周成长周报》。需整合多方信息，提炼核心观点，避免单纯复述，要求知识密度高。

    【执行逻辑】
    1. **提炼高光 (Highlights)**：依据架构与性能评估，识别代码中的亮点、水平或相对于旧档案的技术突破，以及找出优化提示点（如可以运用更加高效的算法或者架构）。
    2. **聚焦改进 (Focus Area)**：从审计报告中筛选出优先级高的 2-3 个问题（如严重安全漏洞、核心逻辑谬误或恶劣的编码习惯），作为本周整改重点。
    3. **全量纠错 (Error Analysis)**：综合审计员与架构师的发现，并结合你对原始代码的审查，罗列代码中存在的逻辑错误与性能低下的片段。
    4. **答疑 (Q&A)**：若[学生心得]中包含具体技术困惑或提问，提供简明解答；若无提问，则跳过此步骤。
    5. **规划下一步 (Next Step)**：针对本周暴露的短板，布置具体的专项训练题目或推荐学习内容（题目，或者官方文档），可以参考【代码审计报告】中的链接与知识点。

    【输出格式】
    严格遵循 Markdown 格式，语气专业且具有指导性，包含以下板块：
    - ** 本周高光**
    - ** 效率改进** (指出部分可能导致代码效率低下的地方，列出用户的代码（效率低下的代码，比如时间复杂度爆炸的代码）与优化后的代码（效率高的解法与算法）)(主要部分，要求输出长，尽量贴近输出上限)
    - ** 错误清单** (指出所有具体错误，列出用户错误的代码与修正后的代码（根据情况提供多种解法）)(主要部分，要求输出长，尽量贴近输出上限)
    - ** 答疑解惑** (若无问题则省略)
    - ** 自身强化** (给出用户下一周可以去学习的部分，比如去看...官方文档，去刷...的题目，给出链接（可以参考【代码审计报告】中的链接（报告中链接较新），也可以根据你的知识库）)
    """

    user_content = user_content = f"""
    【学生心得】: {user_note}
    
    【代码审计报告】:
    {review_res}
    
    【架构评估报告】:
    {architect_res}
    
    【代码片段摘要】:
    {code_snippets}
    """

    model = MODEL_CONFIG["mentor"]
    try:
        async for chunk in call_ai_chat(model,system_prompt,user_content):
            yield chunk
    except Exception as e:
        error_msg = f"\n\n[Architect 运行出错]: {str(e)}"        
        print(error_msg)
        yield error_msg

async def agent_chat(user_query: str):
    """
    [Chat Agent - 随身导师]
    处理多轮对话，自动识别当前是“场景A(带代码)”还是“场景B(纯闲聊)”。
    """
    # 1. 获取当前档案 (无论哪种场景都需要档案)
    current_profile = "暂无档案"
    if os.path.exists(PROFILE_PATH):
        with open(PROFILE_PATH, "r", encoding="utf-8") as f:
            current_profile = f.read()

    # 2. 判断场景
    context_data = st.session_state.current_context
    analysis_res = st.session_state.analysis_result
    
    system_prompt = ""
    user_context_block = ""

    # === 场景 A: 刚刚结束分析，有代码和报告 ===
    if context_data and analysis_res:
        code_text = "\n".join(context_data.get('code', []))[:20000] # 截断防溢出
        mentor_report = analysis_res.get('mentor', '')
        
        system_prompt = f"""
        你是一位严厉但循循善诱的编程导师。你刚刚完成了对该学生代码的周报分析。
        
        【你的主要依据】
        1. **学生档案**: {current_profile}
        2. **刚刚分析的代码**: (见下文)
        3. **你给出的周报**: (见下文)
        
        【回复策略】
        - 既然你手里有代码，当学生提问时，**必须引用具体代码行数**来解释。
        - 结合你刚才指出的错误清单进行回答。
        - 保持多轮对话的连贯性，不要重复自我介绍。
        """
        
        user_context_block = f"""
        【当前代码上下文】:
        {code_text}
        
        【你刚刚生成的周报】:
        {mentor_report}
        """

    # === 场景 B: 刷新后/无代码，只有档案 ===
    else:
        system_prompt = f"""
        你是一位编程导师。目前没有具体的代码上下文，但你了解这位学生的历史能力。
        
        【学生档案】: 
        {current_profile}
        
        【回复策略】
        - 回答关于编程、职业规划或技术概念的通用问题。
        - 如果学生问具体的代码细节，请礼貌地告知需要先上传代码进行分析。
        - 根据档案中的“当前弱点”提供针对性的建议。
        """
        
        user_context_block = "【当前状态】: 无代码上下文，仅基于档案交流。"

    # 3. 构建历史对话上下文 (为了让AI由记忆)
    # 将最近15 轮对话拼接成文本传给 AI，模拟记忆
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history[-15:]])
    
    final_user_content = f"""
    {user_context_block}
    
    【历史对话回顾】:
    {history_text}
    
    【学生当前提问】: 
    {user_query}
    """

    model = MODEL_CONFIG["reviewer"]

    async for chunk in call_ai_chat(model, system_prompt, final_user_content):
        yield chunk

# ==========================================
# 4. 主工作流控制 (Workflow)
# ==========================================

# async def run_weekly_analysis(uploaded_files, user_note, current_profile):

    
async def main():
    # 1. 必须最先执行配置
    st.set_page_config(page_title="AI Coding Mentor", layout="wide", page_icon="🧙‍♂️")
    
    # 2. CSS 样式优化
    st.markdown("""
    <style>
    .stTextArea textarea { font-size: 16px; }
    div[data-testid="stExpander"] details summary p { font-size: 1.1rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None  

    #用于存储多轮对话历史(刷新后)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    # 用于持久化存储解析后的代码和文档内容
    if "current_context" not in st.session_state:
        st.session_state.current_context = None

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("🧙‍♂️ 个人档案")
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                profile_content = f.read()
                with st.expander("📜 点击查看完整档案", expanded=False):
                    st.markdown(profile_content)
        else:
            st.warning("暂无档案，请先进行一次周报分析。")

        st.divider()
        if st.button("🗑️ 清除所有数据"):
            if os.path.exists(PROFILE_PATH): os.remove(PROFILE_PATH)
            if os.path.exists(HISTORY_PATH): os.remove(HISTORY_PATH)
            st.session_state.analysis_result = None
            st.rerun()

    # --- 主界面 ---
    st.title("AI Coding Mentor")
    st.caption("你的私人技术成长顾问团队")

    # 使用 Tabs 分离工作台与历史
    tab_analysis, tab_chat, tab_history = st.tabs(["🚀 本周分析", "💬 导师对话", "📜 历史档案"])
    # ==========================
    # Tab 1: 分析工作台
    # ==========================
    with tab_analysis:
        col_input, col_note = st.columns([1, 1])
        with col_input:
            uploaded_files = st.file_uploader("1. 上传代码/文档", accept_multiple_files=True)
        with col_note:
            user_note = st.text_area("2. 本周心得", height=100, placeholder="例如：这周主要学习了...")

        # 【修改点 1】在这里创建一个空的容器占位符，位置在按钮上方
        status_placeholder = st.empty()

        start_btn = st.button("启动周报分析", type="primary", use_container_width=True)
        st.divider()

        # 预先定义布局容器（防止UI跳动）
        st.subheader("第一阶段：深度技术评估")
        col_review, col_arch = st.columns(2)
        with col_review:
            st.markdown("#### 代码审计 (Reviewer)")
            # 使用 container 固定高度，美观
            review_box = st.container(height=500, border=True)
            review_placeholder = review_box.empty()
        
        with col_arch:
            st.markdown("#### 架构评估 (Architect)")
            arch_box = st.container(height=500, border=True)
            arch_placeholder = arch_box.empty()

        st.subheader("第二阶段：导师总结 (Mentor)")
        mentor_box = st.container(border=True)
        mentor_placeholder = mentor_box.empty()

        # --- 核心逻辑 A: 点击运行 ---
        if start_btn:
            if not uploaded_files:
                st.error("⚠️ 请先上传文件！")
            else:
                # 【修改点 2】指定在这个占位符容器内渲染 st.status
                with status_placeholder:
                    # 使用 st.status 显示进度状态
                    with st.status("🔥 AI 团队正在并行工作中...", expanded=True) as status:
                        
                        async def run_async_logic():
                            try:
                                # 1. Librarian
                                st.write("Librarian: 正在整理文件并更新档案...")
                                context, _ = await agent_librarian(uploaded_files)
                                st.session_state.current_context = context
                                await agent_librarian_write(context['code']) 

                                # 2. Reviewer & Architect 并行
                                st.write("Reviewer & Architect: 正在分析代码...")
                                
                                # 临时存储结果用于显示
                                results = {"review": "", "arch": "", "mentor": ""}

                                # 定义流式回调
                                async def stream_review():
                                    async for chunk in agent_reviewer(context):
                                        results["review"] += chunk
                                        review_placeholder.markdown(results["review"] + "▌")
                                    review_placeholder.markdown(results["review"])

                                async def stream_arch():
                                    async for chunk in agent_architect(context):
                                        results["arch"] += chunk
                                        arch_placeholder.markdown(results["arch"] + "▌")
                                    arch_placeholder.markdown(results["arch"])

                                await asyncio.gather(stream_review(), stream_arch())

                                # 3. Mentor
                                st.write("Mentor: 正在撰写周报...")
                                async for chunk in agent_mentor(results["review"], results["arch"], user_note, context):
                                    results["mentor"] += chunk
                                    mentor_placeholder.markdown(results["mentor"] + "▌")
                                mentor_placeholder.markdown(results["mentor"])

                                # 4. 保存状态与文件
                                st.session_state.analysis_result = results
                                
                                new_record = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "note": user_note,
                                    **results 
                                }
                                
                                history = []
                                if os.path.exists(HISTORY_PATH):
                                    try:
                                        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                                            history = json.load(f)
                                    except: pass
                                
                                history.append(new_record)
                                with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                                    json.dump(history, f, ensure_ascii=False, indent=2)

                                status.update(label="✅ 分析完成！已归档", state="complete", expanded=False)
                                st.balloons()
                                
                            except Exception as e:
                                st.error(f"运行出错: {e}")

                        
                        await run_async_logic()

        # --- 核心逻辑 B: 回填旧数据 (防止刷新白屏) ---
        elif st.session_state.analysis_result:
            res = st.session_state.analysis_result
            review_placeholder.markdown(res["review"])
            arch_placeholder.markdown(res["arch"])
            mentor_placeholder.markdown(res["mentor"])

    # ==========================
    # 历史档案
    # ==========================
    with tab_history:
        if not os.path.exists(HISTORY_PATH):
            st.info("📭 暂无历史记录")
        else:
            try:
                with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 倒序遍历（最新的在最前）
                for idx, item in enumerate(reversed(data)):
                    ts = item.get('timestamp', 'Unknown')
                    note = item.get('note', '')[:30]
                    
                    with st.expander(f"📅 {ts} | 心得: {note}...", expanded=(idx==0)):
                        t1, t2, t3 = st.tabs(["导师周报", "代码审计", "架构评估"])
                        with t1: st.markdown(item.get('mentor', ''))
                        with t2: st.markdown(item.get('review', ''))
                        with t3: st.markdown(item.get('arch', ''))
            except Exception as e:
                st.error(f"历史记录读取失败: {e}")
    # ==========================
    # 导师对话
    # ==========================
    with tab_chat:
        # 1. 顶部状态提示 (可选，放在最上面)
        if st.session_state.current_context:
            st.success("🧠 已连接代码大脑：AI 已读取你刚刚提交的代码和报错，可直接提问。")
        else:
            st.info("💬 闲聊模式：AI 仅了解你的历史档案，无当前代码数据。")

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("向导师提问 (例如：这行代码为什么报错？)"):
            
            # A. 用户提问立即显示 (追加在历史记录下方)
            with st.chat_message("user"):
                st.markdown(prompt)
            # 更新历史数据
            st.session_state.chat_history.append({"role": "user", "content": prompt})

            # B. AI 回复 (流式显示)
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                
                # 调用 agent_chat 生成回复
                async def stream_chat():
                    nonlocal full_response
                    async for chunk in agent_chat(prompt):
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")
                    response_placeholder.markdown(full_response)
                
                await stream_chat()
            
            # C. 保存 AI 回复到历史
            st.session_state.chat_history.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    asyncio.run(main())