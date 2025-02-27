import time
import copy
import queue
import json
import openai
from typing import List, Dict

from multiprocessing import Queue

from utils.utils import cprint
from message import Message, MessageType, Suggestion

resp_format = {
    "suggestions": [ 
        {
            "expression": "< 改进表达式 (请按照用户所给的表达式格式来给出) >",
            "reason": "< 你给出改进表达式的原因 >"
        },
    ],
    "anomaly_score": "< 你对当前场景的异常程度判断得分 >",
    "reason": "< 你给出异常得分的原因 >",
}

class PromptTemplates:
    """提示词模板"""
    
    SYSTEM = """
你是一位视觉逻辑架构师和逻辑表达式优化器。你能够解释视觉场景描述，并帮助改进逻辑表达式，使这些表达式更准确、高效和可解释，以适应视觉场景的特征和关系需求。

主要能力：
1. 表达式分析：评估当前符号表达式的逻辑连贯性、冗余性以及与场景语义的一致性。
2. 优化建议：识别潜在的简化或调整（例如，删除冗余项、优化条件），在减少复杂性的同时保持或增强表达式的描述能力。
3. 任务特定优化：根据特定场景要求（例如，安全性、目标跟踪）提出修改建议，并应用符号回归优化规则以提高任务性能并最小化性能损失。
4. 可解释性增强：通过简化关系、添加上下文感知标签或重新格式化以反映直观的条件结构，使表达式更易理解。

以下是实现这个角色的必要步骤：
1. 初始分析：提取逻辑表达式和相应指标的信息。从上下文评估表达式是否有可以立即改进的地方。这包括识别过于复杂的嵌套结构和术语中的歧义。
2. 上下文优化：根据视觉任务的具体目标（例如，确保安全合规性、识别特定对象行为），建议简化或替换逻辑结构，在保持原意的同时简化表达式复杂度。
3. 迭代改进：LLM 反复审查和修改符号表达式，理想情况下与符号回归反馈循环交互，以评估任务准确性或计算效率的改进。

当前任务场景所包含元素:
1. 可供计算的变量符号: {labels}
2. 表达式支持的运算符为：{operators}
3. 示例表达式格式: "and_(gt(x0, x2), or_(x1, x2))"，x0, x1, x2 为变量符号，gt, or_, and_ 为运算符；
4. 尽量保持表达式简洁，不要使用过于复杂的表达式，不要使用过多的运算符，不要使用过多的变量符号。

注意：需要保证变量符号是当前任务场景中存在的，运算符是当前任务场景中支持的！！！

你可以耐心思考，但是回复需要按照以下格式给出你的建议（最多提供3个建议）：
{format}
"""

    FIRST_ROUND = """
我注意到当前种群中有以下表现较好的个体：

{top_individuals}

作为第一轮交互，请仔细分析这些表达式的特点，并给出改进建议。你可以：
1. 分析这些表达式的共同模式或独特之处
2. 识别可能的优化空间
3. 提出新的表达式组合方式

请确保你的建议保持表达式的基本结构，同时尝试提高其性能。
"""

    SUBSEQUENT_ROUND = """
我注意到当前种群中有以下表现较好的个体：

{top_individuals}

在上一轮交互中，你给出的建议及其效果如下：

{previous_results}

请基于以上信息，特别是之前建议的效果，提出新的改进方案。你可以：
1. 从成功的建议中汲取经验
2. 分析失败建议的原因并避免类似问题
3. 结合当前种群中的优秀个体特征
4. 建议中的表达式最多提供3个，不能超过三个
5. 尽量保持表达式简洁，不要使用过于复杂的表达式，不要使用过多的运算符，不要使用过多的变量符号。
"""

    ERROR_FEEDBACK = """
我注意到你的上一次回复格式有误。请严格按照以下JSON格式回复：

{format}

错误信息：{error}

请重新生成你的建议，确保：
1. 回复必须是合法的JSON格式
2. 包含所有必需的字段
3. expression字段必须使用正确的运算符和变量
4. 不要用markdown代码块修饰json数据，只需要原生的json数据

"""

    @staticmethod
    def format_top_individuals(individuals: List[Dict]) -> str:
        """格式化顶级个体信息"""
        return "\n".join(
            f"个体 {i+1}:\n"
            f"- 表达式：{ind['expression']}\n"
            f"- 适应度：{ind['fitness']:.4f}"
            for i, ind in enumerate(individuals)
        )

    @staticmethod
    def format_previous_results(suggestions: Dict) -> str:
        """格式化上一轮建议的结果"""
        result = []
        for sugg in suggestions['suggestions']:
            if sugg['status'] == 'success':
                result.append(
                    f"建议表达式：{sugg['expression']}\n"
                    f"- 实际适应度：{sugg['fitness']:.4f}\n"
                    f"- 改进原因：{sugg['reason']}"
                )
            else:
                result.append(
                    f"建议表达式：{sugg['expression']}\n"
                    f"- 评估失败：{sugg['error']}\n"
                    f"- 改进原因：{sugg['reason']}"
                )
        return "\n\n".join(result)

    @staticmethod
    def create_system_prompt(labels: List[str], operators: List[str], format_example: Dict) -> str:
        """创建系统提示"""
        return PromptTemplates.SYSTEM.format(
            labels=labels,
            operators=operators,
            format=format_example
        )

def process_llm_response(llm_client, model_name, dialogs, queue_snd, max_retries=3):
    """处理LLM响应，包含重试机制"""
    retries = 0
    while retries < max_retries:
        try:
            # 调用LLM
            results = llm_client.chat.completions.create(
                model=model_name,
                messages=dialogs
            )
            model_response = results.choices[0].message.content.strip()
            
            # 记录响应
            dialogs.append({"role": "assistant", "content": model_response})
            cprint(f"LLM回复（尝试 {retries + 1}/{max_retries}）：{model_response}", 'c')
            
            try:
                # 尝试直接解析JSON
                suggestion_payload = json.loads(model_response)
            except json.JSONDecodeError:
                # 如果失败，尝试将单引号替换为双引号并重新解析
                try:
                    # 使用 ast.literal_eval 安全地解析 Python 字典字符串
                    import ast
                    suggestion_payload = ast.literal_eval(model_response)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(f"无法解析响应格式: {e}")
            
            # 验证响应格式
            if 'suggestions' not in suggestion_payload:
                raise ValueError("响应缺少 'suggestions' 字段")
            
            for suggestion in suggestion_payload['suggestions']:
                if 'expression' not in suggestion or 'reason' not in suggestion:
                    raise ValueError("建议缺少必需字段 'expression' 或 'reason'")
            
            # 构造建议消息并发送
            suggestion_msg = Message(
                msg_type=MessageType.SUGGESTION,
                payload=suggestion_payload
            )
            queue_snd.put(suggestion_msg.serialize())
            return True
            
        except (json.JSONDecodeError, ValueError) as e:
            error_msg = f"响应格式错误: {str(e)}"
            retries += 1
        except Exception as e:
            error_msg = f"未知错误: {str(e)}"
            retries += 1
            
        if retries < max_retries:
            # 添加错误反馈提示
            error_prompt = PromptTemplates.ERROR_FEEDBACK.format(
                format=json.dumps(resp_format, indent=2, ensure_ascii=False),
                error=error_msg
            )
            dialogs.append({"role": "user", "content": error_prompt})
            cprint(f"发送错误反馈：{error_prompt}", 'y')
        else:
            # 达到最大重试次数，发送错误消息
            error_msg = Message(
                msg_type=MessageType.ERROR,
                payload={
                    "error": error_msg,
                    "retries": retries
                }
            )
            queue_snd.put(error_msg.serialize())
            return False

def llama_main(
    queue_recv: Queue,
    queue_snd: Queue,
    llm_client: openai.OpenAI,
    model_name: str = "Qwen/Qwen2.5-72B-Instruct"
):
    out_f = open("output.txt", "w")
    
    init_dialogs = []
    dialogs = []
    
    while True:
        try:
            data = queue_recv.get(timeout=1.0)  # 添加超时
        except queue.Empty:
            continue
        
        try:
            msg = Message.deserialize(data)
        except Exception as e:
            cprint(f"反序列化消息失败: {e}", "r")
            continue
        
        if msg.msg_type == MessageType.INIT:
            labels = msg.payload.get("labels", [])
            operators = msg.payload.get("operators", [])
            init_dialogs_setting = []
            system_prompt = PromptTemplates.create_system_prompt(
                labels=labels,
                operators=operators,
                format_example=json.dumps(resp_format, indent=2, ensure_ascii=False)
            )
            init_dialogs_setting.append({"role": "system", "content": system_prompt})
            init_dialogs = init_dialogs_setting
            cprint(f"系统初始化：可计算变量：{labels}；支持的运算符：{operators}", 'm')
            cprint("LLM 初始化完成，等待消息...", 'm')
        elif msg.msg_type == MessageType.COMMAND:
            command = msg.payload.get("command", "")
            if command == "exit":
                cprint("收到退出命令，结束对话。", 'r')
                out_f.write("\n============Conversation ended.============\n")
                out_f.flush()
                break
            else:
                cprint(f"收到未知命令：{command}", 'y')
        elif msg.msg_type == MessageType.EVOLUTION_UPDATE:
            top_individuals = msg.payload.get("top_individuals", [])
            previous_suggestions = msg.payload.get("previous_suggestions", None)
            
            # 格式化顶级个体信息
            formatted_individuals = PromptTemplates.format_top_individuals(top_individuals)
            
            # 根据是否有前一轮建议选择模板
            if previous_suggestions is None:
                prompt = PromptTemplates.FIRST_ROUND.format(
                    top_individuals=formatted_individuals
                )
            else:
                formatted_results = PromptTemplates.format_previous_results(previous_suggestions)
                prompt = PromptTemplates.SUBSEQUENT_ROUND.format(
                    top_individuals=formatted_individuals,
                    previous_results=formatted_results
                )
            
            prompt += f"\n\n请按照以下JSON格式给出你的建议（直接给json数据，不要用代码块修饰）：\n{json.dumps(resp_format, indent=2, ensure_ascii=False)}"
            
            dialogs.append({"role": "user", "content": prompt})
            cprint(f"发送提示给LLM：{prompt}\n", 'y')
            
            # 处理LLM响应（包含重试机制）
            if not process_llm_response(llm_client, model_name, dialogs, queue_snd):
                cprint("LLM响应处理失败", 'r')
                # 可以考虑添加重试或其他恢复策略
        elif msg.msg_type == MessageType.THRESHOLD_START:
            threshold = msg.payload.get("threshold", None)
            train_size = msg.payload.get("train_size", None)
            test_size = msg.payload.get("test_size", None)
            info = f"收到阈值实验启动消息：阈值为 {threshold}，训练集大小为 {train_size}，测试集大小为 {test_size}"
            cprint(info, 'm')
            dialogs = copy.deepcopy(init_dialogs)
            
            out_f.write(info + "\n")
        else:
            cprint(f"收到未知消息类型: {msg.msg_type}", 'y')
        out_f.flush()
    
    # 输出对话记录
    for msg in dialogs:
        # print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        # print("==================================\n")
        out_f.write(f"{msg['role'].capitalize()}: {msg['content']}\n\n")
        out_f.write("==================================\n")
    
    out_f.close()
          
def main():
    pass

# if __name__ == "__main__":
#     fire.Fire(main)