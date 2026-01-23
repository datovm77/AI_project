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

# streamlit run agent1.0.py
# ==========================================
# 1. ⚙️ 配置与初始化
# ==========================================
load_dotenv()  #导入secrets

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
    输出: 字典 {'filename':..., 'type': 'code'/'document'/'image', 'content':...}
    """
    file_type = uploaded_file.name.split('.')[-1].lower()
    result = {
        "filename":uploaded_file.name,
        "type":"unknow",
        "content":""
    }
    text = ""
    try:
        if file_type == 'pdf':
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages: 
                    text += page.text + '\n'
            result["type"] = "document"
            result["content"] = text    
        elif file_type == 'docx':
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs: text += para.text + "\n"
            result["type"] = "document"
            result["content"] = text
        elif file_type in ['txt', 'c', 'cpp', 'py', 'java', 'md']:
            text = uploaded_file.read().decode("utf-8", errors='ignore')
            result["type"] = "code"
            result["content"] = text
        elif file_type in ['png', 'jpg', 'jpeg']:
            bytes_data = uploaded_file.getvalue()
            text = encode_image_to_base64(bytes_data)
            result["type"] = "image"
            result["content"] = text
    except Exception as e:
        return f"[读取出错: {str(e)}]"
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
    你是一位极具洞察力的“技术档案管理员”。
    你的任务是根据【旧档案】和用户本周提交的【原始代码】，**推断**用户的技术成长，并生成一份**更新后的档案**。

    【更新逻辑】：
    1. **技能捕获**：不要听用户说什么，要看代码里用了什么。发现了新的库(Library)、新的语法特性或设计模式吗？加入技能树。
    2. **代码品味**：观察代码风格。是有详尽注释、模块化良好？还是充满了硬编码和意大利面条代码？据此调整“代码质量”或“弱点”字段。
    3. **动态评级**：如果代码逻辑复杂且优雅，升级评价(S/A/B/C)；如果全是低价水平，保持或降级。
    4. **只输出档案**：直接输出更新后的完整档案内容（Markdown格式），不需要任何开场白。
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
    你是一位拥有10年经验的**高级代码审计员 (Code Reviewer)**。
    你的工作风格是：严谨、犀利、关注细节，绝不放过任何一个安全隐患。

    【任务目标】
    请仔细阅读用户提供的【代码片段】以及可能的【报错截图/运行截图】，生成一份审计报告。

    【关注重点】
    1. **安全性 (Security)**：是否存在 SQL 注入、XSS、硬编码密钥、敏感信息泄露等风险？
    2. **健壮性 (Robustness)**：是否存在未捕获的异常、空指针引用、死循环风险？
    3. **Bug 分析 (Diagnostics)**：如果输入包含图片（报错截图），请优先分析报错原因，并指出代码中对应的错误行。
    4. **代码异味 (Code Smells)**：是否存在命名混乱、魔法数字、冗余逻辑？

    【输出格式要求】
    请使用 Markdown 格式，结构如下：
    - **🔴 致命问题**：(会导致崩溃或严重安全漏洞的问题，无则不写)
    - **🟡 改进建议**：(性能优化、逻辑简化)
    - **📸 截图分析**：(如果有图片，简述报错含义；无图片则忽略此项)
    - **💡 修复代码片段**：(仅针对最严重的问题给出简短的修复示例)

    请保持客观冷静，直接切入技术点，不要说废话。
    """
    code_snippets = context.get('code',[])
    image_list = context.get('image',[])

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
    你是一位眼光长远的**首席技术架构师 (Chief Architect)**。
    你的工作风格是：宏观、战略性、注重代码的可维护性和设计美感。
    你需要结合用户的【旧档案 (Old Profile)】和【本周代码】，评估其技术成长。

    【任务目标】
    1. **设计模式识别**：代码中是否使用了面向对象设计、函数式编程、或特定的设计模式（单例、工厂、观察者等）？
    2. **复杂度评估**：代码是简单的脚本堆砌，还是具有模块化、分层架构？(S/A/B/C 评级)。
    3. **成长性对比 (关键)**：
    - 对比【旧档案】中的技能水平，本周的代码是否有突破？
    - 用户是在重复造轮子（停滞），还是在尝试新技术（成长）？
    4. **技术栈分析**：识别代码中用到的关键库或框架。

    【输出格式要求】
    请使用 Markdown 格式，结构如下：
    - **🏗️ 架构视点**：(评价代码结构、模块化程度)
    - **📈 成长评估**：(明确指出相比旧档案，本周是"突破"、"巩固"还是"停滞")
    - **🛠️ 技术栈侦测**：(列出检测到的关键技术/库)
    - **⚖️ 综合评级**：给出本周代码的综合评分 (S/A/B/C) 并简述理由。

    请不要纠结于具体的语法错误（那是审计员的事），你要关注的是“代码的品味”和“开发者的上限”。
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
    你是一位**技术导师 (Tech Mentor)**，你的学生刚提交了本周的代码。
    你手头有两份技术报告：
    1. **代码审计员 (Reviewer)**：指出了具体的 Bug 和安全隐患。
    2. **架构师 (Architect)**：评估了设计模式和技术成长。
    3. **学生心得 (User Note)**：学生自己写的本周感悟。
    4. **学生原始代码**:学生本周写的代码
    【任务目标】
    请你用**耐心专业**的口吻，写一份《本周成长周报》。

    【内容结构】
    1. **本周高光 (Highlights)**：结合架构师的评价，表扬做得好的地方。
    2. **核心改进 (Focus Area)**：综合审计员的报告，指出下周最需要集中精力解决的 1-2 个坏习惯或技术短板。
    3. **错误说明 (show mistake)** 综合架构师与审计师（主）与自己对代码的理解（辅），指出所有（所有）的错误。
    3. **答疑解惑 (Q&A)**：如果学生的【学习心得】里提出了问题或困惑，请简要解答；如果没有，则忽略此项。
    4. **下周挑战 (Next Step)**：根据当前水平，布置几个专项训练（可以是题目或者是某一个知识点）或推荐一个学习关键词。

    请避免直接的重复前两份报告的内容，而是要提炼核心观点，找出所有可能的错误，转化为易于消化的建议。
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
        yield f"[Mentor 运行出错]: {str(e)}"


# ==========================================
# 4. 主工作流控制 (Workflow)
# ==========================================

async def run_weekly_analysis(uploaded_files, user_note, current_profile):
    """
    主控函数
    """
    # TODO:
    # Step 1: await agent_librarian(...) -> 得到 structured_context
    # Step 2: asyncio.gather(agent_reviewer(...), agent_architect(...)) -> 并发获取两份报告
    # Step 3: await agent_mentor(...) -> 得到最终周报
    # Return: final_report
    


# ==========================================
# 5. UI 入口 (Main)
# ==========================================

def main():
    st.set_page_config(page_title="AI Coding Mentor", layout="wide", page_icon="🧙‍♂️")
    
    # --- CSS 样式优化 (可选) ---
    st.markdown("""
    <style>
    .stTextArea textarea { font-size: 16px; }
    div[data-testid="stExpander"] details summary p { font-size: 1.1rem; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)


    with st.sidebar:
        st.header("你的个人档案")
        
        # 实时读取档案
        current_profile_content = "暂无档案"
        if os.path.exists(PROFILE_PATH):
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                current_profile_content = f.read()
            st.info(current_profile_content)
        else:
            st.warning("欢迎新人！完成第一次周报分析后将自动生成档案。")

        st.divider()
        with st.expander("历史记录管理"):
            st.caption(f"存储路径: `{HISTORY_PATH}`")
            if st.button("清除所有历史 & 档案"):
                if os.path.exists(PROFILE_PATH): os.remove(PROFILE_PATH)
                if os.path.exists(HISTORY_PATH): os.remove(HISTORY_PATH)
                st.success("重置成功！")
                st.rerun()

    # --- 主界面 ---
    st.title("AI Coding Mentor ")
    st.markdown("### 你的私人技术成长顾问团队")
    st.caption("上传本周代码，AI 团队将并行工作：Librarian 整理档案 -> Reviewer 审计代码 -> Architect 评估架构 -> Mentor 生成周报")

    # --- 1. 输入区域 ---
    col_input, col_note = st.columns([1, 1])
    with col_input:
        uploaded_files = st.file_uploader("1. 上传代码文件 (支持 .py, .java, .cpp, .pdf, 图片等)", accept_multiple_files=True)
    with col_note:
        user_note = st.text_area("2. 本周心得 / 遇到的困难", height=150, placeholder="例如：这周深入学习了异步编程，但在错误处理上还有点懵...")

    # --- 2. 执行逻辑 ---
    if st.button("启动周报分析", type="primary", use_container_width=True):
        if not uploaded_files:
            st.error("请先上传至少一个文件！")
            return
        
        # --- UI 布局准备 ---
        st.divider()
        status_container = st.status("AI 团队集结中...", expanded=True)
        
        # 创建两列用于并行展示技术分析
        st.subheader("第一阶段：深度技术评估")
        col_review, col_arch = st.columns(2)
        
        with col_review:
            st.markdown("#### 代码审计报告 (Reviewer)")
            reviewer_box = st.container(height=500, border=True)
            reviewer_placeholder = reviewer_box.empty()
            
        with col_arch:
            st.markdown("#### 架构评估报告 (Architect)")
            architect_box = st.container(height=500, border=True)
            architect_placeholder = architect_box.empty()

        st.subheader("第二阶段：导师总结周报 (Mentor)")
        mentor_box = st.container(border=True)
        mentor_placeholder = mentor_box.empty()

        # --- 核心异步流程 (这就是原本的 run_weekly_analysis) ---
        async def run_loop():
            try:
                # Step 1: Librarian 整理文件
                status_container.write("Librarian: 正在解析并分类上传的文件...")
                context, _ = await agent_librarian(uploaded_files)
                
                # Step 2: Librarian 更新档案 (后台静默更新)
                status_container.write("Librarian: 正在对比历史档案并更新能力树...")
                # 注意：这里我们让它并行跑，还是阻塞跑？为了后续 Architect 能读到最新对比，建议先跑完，或 Architect 读旧的。
                # 逻辑选择：Architect 读旧档案做对比更有意义（对比上周 vs 本周）。
                # 所以我们让 Profile 更新在后台进行，或者最后进行。这里选择先计算出新 Profile 内容备用。
                _ = await agent_librarian_write(context['code']) 
                status_container.write("档案已更新 (Architect 将基于旧档案对比成长)")

                # Step 3: 并行执行 Reviewer 和 Architect
                status_container.write("Reviewer & Architect: 正在并行分析代码...")
                
                # 定义用于流式更新 UI 的内部函数
                reviewer_res = ""
                architect_res = ""

                async def stream_reviewer():
                    nonlocal reviewer_res
                    async for chunk in agent_reviewer(context):
                        reviewer_res += chunk
                        reviewer_placeholder.markdown(reviewer_res + "▌")
                    reviewer_placeholder.markdown(reviewer_res) # 结束时去掉光标

                async def stream_architect():
                    nonlocal architect_res
                    async for chunk in agent_architect(context):
                        architect_res += chunk
                        architect_placeholder.markdown(architect_res + "▌")
                    architect_placeholder.markdown(architect_res)

                # 并发启动！
                await asyncio.gather(stream_reviewer(), stream_architect())

                # Step 4: Mentor 汇总
                status_container.write(" Mentor: 正在阅读技术报告并撰写周报...")
                mentor_res = ""
                async for chunk in agent_mentor(reviewer_res, architect_res, user_note, context):
                    mentor_res += chunk
                    mentor_placeholder.markdown(mentor_res + "▌")
                mentor_placeholder.markdown(mentor_res)

                # Step 5: 完成与存档
                status_container.update(label="本周分析已完成！", state="complete", expanded=False)
                
                # 保存历史
                new_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note": user_note,
                    "review": reviewer_res,
                    "architecture": architect_res,
                    "mentor": mentor_res
                }
                
                history_list = []
                if os.path.exists(HISTORY_PATH):
                    try:
                        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                            history_list = json.load(f)
                    except: pass
                
                history_list.append(new_record)
                with open(HISTORY_PATH, "w", encoding="utf-8") as f:
                    json.dump(history_list, f, ensure_ascii=False, indent=2)

                st.balloons()
                st.toast("周报已保存至 history.json", icon="💾")
                
                # 延迟刷新以显示最新的 Profile
                await asyncio.sleep(3)
                st.rerun()

            except Exception as e:
                st.error(f"运行过程中发生错误: {str(e)}")
                print(e)

        # 启动异步循环
        asyncio.run(run_loop())

if __name__ == "__main__":
    main()