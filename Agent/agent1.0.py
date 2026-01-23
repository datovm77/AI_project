import streamlit as st
import asyncio
import nest_asyncio
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

# streamlit run agent1.0.py
# ==========================================
# 1. ⚙️ 配置与初始化
# ==========================================
load_dotenv()  #导入secrets
nest_asyncio.apply()  # 允许嵌套事件循环

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
                    formatted_report += f"```\n{code[:500]}...\n```\n"
            
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

async def agent_librarian(uploaded_files) -> Dict[str, Any]:
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
    根据提供的[旧档案]与[本周原始代码]，全量生成一份[更新后的技术档案]。

    【处理逻辑】
    1. **技能提取 (基于事实)**：扫描代码中实际使用的库 (Libraries)、框架、语法特性及设计模式。若发现新技能，将其合并不重复地加入技能树。
    2. **质量画像更新**：分析代码的工程质量（注释规范、命名风格、模块化程度、硬编码情况）。据此客观修正“代码风格”与“当前弱点”字段。
    3. **动态评级**：依据本周代码的逻辑复杂度与健壮性，动态调整综合技术评级 (S/A/B/C)。
    4. **录入删减规则**：不得随意删减旧档案内容。档案的内容重复时，根据情况可以对档案做出适当修改。
    【输出约束】
    1. 格式必须为 Markdown。
    2. **严禁输出**任何开场白、解释语或结束语。
    3. **直接输出**完整的档案内容。
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
    职责：安全审计、Bug 查找、报错分析。
    【多模态需求】：高 (需要看报错截图)
    """

    system_prompt = """
    【任务指令】
    对用户提供的[代码片段]及[运行截图]执行安全与健壮性审计。

    【执行逻辑】
    1. **视觉诊断 **：若包含图片（报错/运行截图），优先解析错误信息，并定位代码中的具体致错行。
    2. **安全扫描**：检测关键漏洞（SQL注入、XSS、硬编码密钥、敏感数据泄露、越权访问）。
    3. **健壮性评估**：识别运行时风险（空指针、未捕获异常、死循环、资源未关闭、语法错误）。
    4. **代码异味**：指出不可读命名、魔法数字、冗余逻辑或反模式写法。

    【输出格式】
    必须使用 Markdown 格式，仅包含以下板块（若某板块无内容则省略）：
    - **🔴 致命问题**：(会导致崩溃或严重安全漏洞的问题)
    - **🟡 改进建议**：(性能优化、逻辑简化、代码规范)
    - **📸 截图分析**：(针对图片中报错信息的简要技术解读)
    - **💡 修复代码**：(仅针对最严重问题提供最小化修复片段)

    【风格约束】
    客观、直接、技术导向。严禁输出寒暄语。
    """
    code_snippets = context.get('code',[])
    image_list = context.get('images',[])

    full_code_text = "\n\n".join(code_snippets)
    if len(full_code_text) > 30000:
        full_code_text = full_code_text[:30000] + "\n\n(代码过长，后续部分已截断...)"

    if not full_code_text and not image_list:
        yield"[审计员]：没有代码文本与截图，本周内容无"
        return
    
    user_content = f"【待处理代码:】:\n{full_code_text}"
    if not full_code_text:
        user_content = "【代码内容】: (无文本，仅分析提供的截图)"

    model = MODEL_CONFIG["reviewer"]

    try:
        async for chunk in call_ai_chat(model,system_prompt,user_content,image_base64_list=image_list):
            yield chunk

    except Exception as e:
        error_msg = f"\n\n[Reviewer运行出错]:{str(e)}"        
        print(error_msg)
        yield error_msg



# 记得在文件头部确保导入： from typing import AsyncGenerator

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
    基于用户[旧档案]与[本周代码/文档]，执行宏观架构评估与技术成长判定。忽略具体语法错误，专注代码的可维护性、设计逻辑、技术上限与运行效率（重点）。

    【执行步骤】
    1. **架构分析**：提取代码的结构模式（如分层、模块化程度）。识别是否应用了特定设计模式（OOP、FP、单例、工厂等），是否优化了运行效率，评估架构性能。
    2. **成长比对**：将本周代码的技术深度与[旧档案]进行对比。
    - 判定状态：**突破**（应用了新概念/新技术）、**巩固**（熟练度提升）或 **停滞**（重复低水平劳动）。
    3. **技术栈提取**：罗列代码中使用的核心框架、第三方库或中间件。
    4. **综合定级**：根据代码的工程复杂度与设计美感或者运行效率，给出 S/A/B/C 评级。

    【输出格式】
    严格遵循 Markdown 格式，仅输出以下四个板块：

    - **🏗️ 架构或性能视点**：(简述代码结构、模块划分及运行效率)
    - **📈 成长评估**：(明确指出与旧档案相比的进步点，判定本周状态)
    - **🛠️ 技术栈侦测**：(列出检测到的关键技术/库)
    - **⚖️ 综合评级**：(给出 S/A/B/C 评分并简述理由)
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
    1. **提炼高光 (Highlights)**：依据架构评估，识别代码中的设计亮点、模式应用或相对于旧档案的技术突破。
    2. **聚焦改进 (Focus Area)**：从审计报告中筛选出优先级高的 2-3 个问题（如严重安全漏洞、核心逻辑谬误或恶劣的编码习惯），作为本周整改重点。
    3. **全量纠错 (Error Analysis)**：综合审计员与架构师的发现，并结合你对原始代码的审查，罗列代码中存在的逻辑错误与技术误区。
    4. **答疑 (Q&A)**：若[学生心得]中包含具体技术困惑或提问，提供简明解答；若无提问，则跳过此步骤。
    5. **规划下一步 (Next Step)**：针对本周暴露的短板，布置具体的专项训练题目或推荐一个核心学习关键词。

    【输出格式】
    严格遵循 Markdown 格式，语气专业且具有指导性，包含以下板块：
    - ** 本周高光**
    - ** 核心改进**
    - ** 错误清单** (指出所有具体错误，列出用户错误的代码与修正后的代码（根据情况提供多种解法）)(主要部分，要求输出长，尽量贴近输出上限)
    - ** 答疑解惑** (若无问题则省略)
    - ** 自身强化** (给出用户下一周可以去学习的部分，比如去看...知识点，去刷...的题目)
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


# ==========================================
# 4. 主工作流控制 (Workflow)
# ==========================================

# async def run_weekly_analysis(uploaded_files, user_note, current_profile):

    
def main():
    # 1. 必须最先执行配置
    st.set_page_config(page_title="AI Coding Mentor", layout="wide", page_icon="🧙‍♂️")
    
    # 2. CSS 样式优化
    st.markdown("""
    <style>
    .stTextArea textarea { font-size: 16px; }
    div[data-testid="stExpander"] details summary p { font-size: 1.1rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    # 3. 初始化 Session State (防止刷新丢失)
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None  

    # --- 侧边栏 ---
    with st.sidebar:
        st.header("🧙‍♂️ 个人档案")
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                st.info(f.read())
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
    tab_analysis, tab_history = st.tabs(["🚀 本周分析", "📜 历史档案"])

    # ==========================
    # Tab 1: 分析工作台
    # ==========================
    with tab_analysis:
        col_input, col_note = st.columns([1, 1])
        with col_input:
            uploaded_files = st.file_uploader("1. 上传代码/文档", accept_multiple_files=True)
        with col_note:
            user_note = st.text_area("2. 本周心得", height=100, placeholder="例如：这周主要学习了...")

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
                # 使用 st.status 显示进度状态
                with st.status("🔥 AI 团队正在并行工作中...", expanded=True) as status:
                    
                    async def run_async_logic():
                        try:
                            # 1. Librarian
                            st.write("Librarian: 正在整理文件并更新档案...")
                            context, _ = await agent_librarian(uploaded_files)
                            await agent_librarian_write(context['code']) # 后台更新档案

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

                    
                    asyncio.run(run_async_logic())

        # --- 核心逻辑 B: 回填旧数据 (防止刷新白屏) ---
        elif st.session_state.analysis_result:
            res = st.session_state.analysis_result
            review_placeholder.markdown(res["review"])
            arch_placeholder.markdown(res["arch"])
            mentor_placeholder.markdown(res["mentor"])

    # ==========================
    # Tab 2: 历史档案
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

if __name__ == "__main__":
    main()