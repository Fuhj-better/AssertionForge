# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Add your own LLM client


import os
import tiktoken
# 从 config.py 导入 FLAGS，用于获取 DeepSeek 相关的配置
from config import FLAGS
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type


class LLMAgent:
    """
    一个简单的 LLM 代理对象，用于封装 LLM 的模型名称和其他参数。
    """
    def __init__(self, model_name: str, **kwargs):
        self.llm_model = model_name
        self.llm_args = kwargs # 存储额外的 LLM 参数

    def __str__(self):
        return f"LLMAgent(model='{self.llm_model}', args={self.llm_args})"

def get_llm(model_name: str, **llm_args) -> LLMAgent:
    """
    根据提供的模型名称和参数，返回一个 LLM 代理对象。
    这个代理对象将传递给 llm_inference 函数。

    :param model_name: LLM 的模型名称（例如 'deepseek-chat', 'gpt-4o'）。
    :param llm_args: 传递给 LLM 代理的其他参数。
    :return: 一个 LLMAgent 实例。
    """
    print(f"--- Initializing LLM Agent ---")
    print(f"Model: {model_name}, Args: {llm_args}")
    return LLMAgent(model_name=model_name, **llm_args)


# # --- DeepSeek 配置 ---
# DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# # 从环境变量中获取 DEEPSEEK_API_KEY，如果不存在，则从 FLAGS 获取
# DEEPSEEK_API_KEY = 'sk-fb83fb494d9245f6ba61d03799a7436d'

# --- DeepSeek 配置 ---
DEEPSEEK_BASE_URL = "https://api.siliconflow.cn/v1"

# 从环境变量中获取 DEEPSEEK_API_KEY，如果不存在，则从 FLAGS 获取
DEEPSEEK_API_KEY = 'sk-fypitehevdxfqvcuecdukzthbyywohhpjcvyusjhhcwgyvxz'

if not DEEPSEEK_API_KEY:
    DEEPSEEK_API_KEY = getattr(FLAGS, 'deepseek_api_key', None)

if not DEEPSEEK_API_KEY:
    raise ValueError("DeepSeek API Key is not set. Please set DEEPSEEK_API_KEY environment variable or deepseek_api_key in config.py.")

# 初始化 DeepSeek 客户端
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)


@retry(stop=stop_after_attempt(5), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
def llm_inference(llm_agent_obj, prompt: str, tag: str, max_tokens=None, temperature=0.7, **kwargs) -> str:
    """
    调用大型语言模型 (LLM) 进行推理。

    :param llm_agent_obj: 包含 LLM 配置信息的对象（例如，从 config.FLAGS 传递过来的 llm_agent）。
                          我们期望它至少有一个 'llm_model' 属性。
    :param prompt: 输入给 LLM 的文本提示。
    :param tag: 用于标识本次 LLM 调用的标签（例如日志记录）。
    :param max_tokens: 生成的最大 token 数。
    :param temperature: 控制生成文本的随机性（0.0-2.0）。
    :param kwargs: 其他传递给 LLM API 的参数。
    :return: LLM 生成的文本响应。
    """
    # 从 llm_agent_obj 中获取模型名称
    model_name = getattr(llm_agent_obj, 'llm_model', 'deepseek-chat')

    print(f"--- LLM Inference Call for tag: {tag} ---")
    print(f"Using Model: {model_name} (via DeepSeek API)")
    print(f"Prompt (first 100 chars): {prompt[:100]}...")

    messages = [
        {"role": "user", "content": prompt}
    ]

    try:
        response = deepseek_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content
        else:
            raise ValueError(f"DeepSeek API response for tag '{tag}' is empty or malformed.")

    except Exception as e:
        print(f"Error during DeepSeek LLM inference for tag '{tag}': {e}")
        if "AuthenticationError" in str(e) or "Incorrect API key" in str(e):
            raise Exception("DeepSeek API Key is incorrect or missing. Please check your DEEPSEEK_API_KEY environment variable.")
        raise # 重新抛出异常，让 tenacity 处理重试


def count_prompt_tokens(prompt: str, model_name: str = 'deepseek-chat') -> int:
    """
    计算给定提示的 token 数量。
    使用 tiktoken 库进行计算。

    :param prompt: 要计算 token 的文本提示。
    :param model_name: 用于指定编码方式的模型名称。对于DeepSeek，可以使用其对应的编码模型或一个近似的OpenAI模型。
                        对于deepseek-chat，使用cl100k_base编码器是一个合理的近似。
    :return: 提示中的 token 数量。
    """
    try:
        # 尝试根据模型名称获取编码器
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception:
        print(f"Warning: Could not get tiktoken encoding for model '{model_name}'. Using 'cl100k_base' fallback.")
        encoding = tiktoken.get_encoding("cl100k_base")

    tokens = encoding.encode(prompt)
    return len(tokens)

# # 导入必要的库
# import os
# import tiktoken
# # 从 config.py 导入 FLAGS，用于获取 DeepSeek 相关的配置
# from config import FLAGS
# from openai import OpenAI
# from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type


# class LLMAgent:
#     """
#     一个简单的 LLM 代理对象，用于封装 LLM 的模型名称和其他参数。
#     """
#     def __init__(self, model_name: str, **kwargs):
#         self.llm_model = model_name
#         self.llm_args = kwargs # 存储额外的 LLM 参数

#     def __str__(self):
#         return f"LLMAgent(model='{self.llm_model}', args={self.llm_args})"

# def get_llm(model_name: str, **llm_args) -> LLMAgent:
#     """
#     根据提供的模型名称和参数，返回一个 LLM 代理对象。
#     这个代理对象将传递给 llm_inference 函数。

#     :param model_name: LLM 的模型名称（例如 'deepseek-chat', 'gpt-4o'）。
#     :param llm_args: 传递给 LLM 代理的其他参数。
#     :return: 一个 LLMAgent 实例。
#     """
#     print(f"--- Initializing LLM Agent ---")
#     print(f"Model: {model_name}, Args: {llm_args}")
#     # 这里你可以根据 model_name 返回不同类型的 LLM 客户端
#     # 但目前我们统一使用 LLMAgent 来封装参数
#     return LLMAgent(model_name=model_name, **llm_args)


# # --- DeepSeek 配置 ---
# # DeepSeek API 的基础 URL
# # 这是 DeepSeek 官方提供的 API 地址。请检查 DeepSeek 官方文档以确认最新的 URL。
# DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# # DeepSeek API Key 可以从环境变量中获取，或者从 FLAGS 中获取
# # 为了安全性，强烈建议使用环境变量。
# # 这里我们演示如何从环境变量和 FLAGS 中获取，优先级：环境变量 > FLAGS
# DEEPSEEK_API_KEY = 'sk-fb83fb494d9245f6ba61d03799a7436d'
# if not DEEPSEEK_API_KEY:
#     # 假设 config.py 中有一个 deepseek_api_key 字段
#     DEEPSEEK_API_KEY = getattr(FLAGS, 'deepseek_api_key', None)

# if not DEEPSEEK_API_KEY:
#     raise ValueError("DeepSeek API Key is not set. Please set DEEPSEEK_API_KEY environment variable or deepseek_api_key in config.py.")

# # 初始化 DeepSeek 客户端
# # 注意：这里我们创建一个新的客户端实例，指向 DeepSeek API
# deepseek_client = OpenAI(
#     api_key=DEEPSEEK_API_KEY,
#     base_url=DEEPSEEK_BASE_URL
# )


# @retry(stop=stop_after_attempt(5), wait=wait_fixed(2), retry=retry_if_exception_type(Exception))
# def llm_inference(llm_agent_obj, prompt: str, tag: str, max_tokens=None, temperature=0.7, **kwargs) -> str:
#     """
#     调用大型语言模型 (LLM) 进行推理。

#     :param llm_agent_obj: 包含 LLM 配置信息的对象（例如，从 config.FLAGS 传递过来的 llm_agent）。
#                           我们期望它至少有一个 'llm_model' 属性。
#     :param prompt: 输入给 LLM 的文本提示。
#     :param tag: 用于标识本次 LLM 调用的标签（例如日志记录）。
#     :param max_tokens: 生成的最大 token 数。
#     :param temperature: 控制生成文本的随机性（0.0-2.0）。
#     :param kwargs: 其他传递给 LLM API 的参数。
#     :return: LLM 生成的文本响应。
#     """
#     # 从 llm_agent_obj 中获取模型名称
#     # DeepSeek 模型名称可能与 OpenAI 不同，例如 'deepseek-chat', 'deepseek-coder' 等
#     model_name = getattr(llm_agent_obj, 'llm_model', 'deepseek-chat') # 更改默认模型为 DeepSeek 的模型
#     model_name='deepseek-chat'
#     print(f"--- LLM Inference Call for tag: {tag} ---")
#     print(f"Using Model: {model_name} (via DeepSeek API)") # 指明是 DeepSeek
#     print(f"Prompt (first 100 chars): {prompt[:100]}...")

#     messages = [
#         {"role": "user", "content": prompt}
#     ]

#     try:
#         # 使用 deepseek_client 进行调用
#         response = deepseek_client.chat.completions.create(
#             model=model_name,
#             messages=messages,
#             max_tokens=max_tokens,
#             temperature=temperature,
#             **kwargs
#         )
#         if response.choices and response.choices[0].message and response.choices[0].message.content:
#             return response.choices[0].message.content
#         else:
#             raise ValueError(f"DeepSeek API response for tag '{tag}' is empty or malformed.")

#     except Exception as e:
#         print(f"Error during DeepSeek LLM inference for tag '{tag}': {e}")
#         if "AuthenticationError" in str(e) or "Incorrect API key" in str(e):
#             raise Exception("DeepSeek API Key is incorrect or missing. Please check your DEEPSEEK_API_KEY environment variable.")
#         raise # 重新抛出异常，让 tenacity 处理重试




# def count_prompt_tokens(prompt: str, model_name: str = 'deepseek-chat') -> int:
#     """
#     计算给定提示的 token 数量。
#     使用 tiktoken 库进行计算。

#     :param prompt: 要计算 token 的文本提示。
#     :param model_name: 用于指定编码方式的模型名称。对于DeepSeek，可以使用其对应的编码模型或一个近似的OpenAI模型。
#                        对于deepseek-chat，使用cl100k_base编码器是一个合理的近似。
#     :return: 提示中的 token 数量。
#     """
#     try:
#         # 尝试根据模型名称获取编码器
#         # 对于DeepSeek，通常其编码方式与OpenAI的cl100k_base相似，或者有其专属编码器。
#         # 这里我们先假设使用cl100k_base作为通用近似，你可以根据DeepSeek官方文档调整。
#         encoding = tiktoken.get_encoding("cl100k_base") # 这是一个OpenAI模型的编码器
#         # 如果 DeepSeek 有自己的 tiktoken 兼容编码器，可以在此处替换
#         # 例如：encoding = tiktoken.encoding_for_model(model_name)
#     except Exception:
#         # 如果模型名无法直接映射到 tiktoken 编码器，则回退到默认编码
#         print(f"Warning: Could not get tiktoken encoding for model '{model_name}'. Using 'cl100k_base' fallback.")
#         encoding = tiktoken.get_encoding("cl100k_base")

#     tokens = encoding.encode(prompt)
#     return len(tokens)

