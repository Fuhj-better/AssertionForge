# AssertionForge\src\command_line_args.py

import argparse
from types import SimpleNamespace
from collections import OrderedDict

# 导入 config 模块
# 注意：这里需要确保 config.py 能够被正确导入
# 如果你在 AssertionForge\src 目录下运行 main.py，
# 并且 config.py 也在 src 目录下，那么 'from config import FLAGS' 应该能工作。
try:
    from config import FLAGS as initial_config_FLAGS
except ImportError:
    # 打印警告，但在实际运行时，如果 config.py 不在 sys.path 中，则会出错
    print("Warning: Could not import FLAGS from config.py. Please ensure config.py is in the Python path.")
    # 如果无法导入，我们提供一个空的 SimpleNamespace，但这可能导致后续参数没有默认值
    initial_config_FLAGS = SimpleNamespace()


def parse_command_line_args():
    """
    解析命令行参数并返回一个 SimpleNamespace 对象。
    这个函数会根据 config.py 中定义的初始 FLAGS 动态添加命令行参数。
    命令行参数会覆盖 config.py 中的默认值。
    """
    parser = argparse.ArgumentParser(description="AssertionForge Command Line Arguments")

    # 动态添加从 config.FLAGS 获取的参数
    # 遍历 initial_config_FLAGS 的所有属性
    for key, value in initial_config_FLAGS.__dict__.items():
        # 跳过内部属性，例如那些以下划线开头的属性
        if key.startswith('_'):
            continue

        # 将布尔值转换为 action='store_true' 或 'store_false'
        if isinstance(value, bool):
            parser.add_argument(f'--{key}', action='store_true' if value is False else 'store_false',
                                default=value,  # 默认值是 config.py 中的值
                                help=f'Enable/disable {key} (default: {value})')
        # 对于列表和字典，argparse 默认会将其解析为字符串，
        # 如果需要更复杂的解析，例如 JSON 格式，需要自定义 type
        # 这里为了简化，我们先假设它们作为字符串传递或者不从命令行修改
        elif isinstance(value, (list, dict)):
            # 对于列表和字典，命令行直接修改可能比较复杂。
            # 常见做法是允许用户传入一个 JSON 字符串，然后在代码中解析。
            # 这里我们暂时不处理它们作为可直接从命令行修改的类型，
            # 或者将其视为字符串类型，用户需要自行确保格式。
            parser.add_argument(f'--{key}', type=str, default=str(value),
                                help=f'Set {key} (default: {value}). For complex types like list/dict, provide as string/JSON.')
        elif isinstance(value, type(None)):
            # None 值可能表示未设置，可以将其设置为字符串类型或者自定义处理
            parser.add_argument(f'--{key}', type=str, default=None,
                                help=f'Set {key} (default: None).')
        else:
            # 对于 int, float, str 等简单类型
            parser.add_argument(f'--{key}', type=type(value), default=value,
                                help=f'Set {key} (default: {value})')

    # 解析参数
    args = parser.parse_args()

    # 将解析后的参数转换为 SimpleNamespace 形式
    parsed_flags = SimpleNamespace(**vars(args))

    # 创建最终的 FLAGS 对象
    final_flags = SimpleNamespace()

    # 1. 从 initial_config_FLAGS (config.py) 中加载所有默认值
    for key, value in initial_config_FLAGS.__dict__.items():
        setattr(final_flags, key, value)

    # 2. 用命令行参数覆盖 config 中的值
    # 注意：argparse 会将所有参数都解析出来，包括那些在 config.py 中没有的，
    # 或者类型发生变化的（例如将列表/字典解析为字符串）
    for key, value in parsed_flags.__dict__.items():
        # 对于从命令行解析的字符串，如果它原本是列表或字典，我们尝试转换回去
        if isinstance(getattr(initial_config_FLAGS, key, None), (list, dict)) and isinstance(value, str):
            try:
                import json
                setattr(final_flags, key, json.loads(value))
            except json.JSONDecodeError:
                # 如果解析失败，则保留为字符串
                setattr(final_flags, key, value)
        # 对于布尔值，argparse 的 store_true/store_false 已经处理好了
        # 对于其他类型，直接覆盖
        else:
            setattr(final_flags, key, value)

    # # 针对文件路径 /<path>/<to>/ 的替换逻辑
    # # 这是 config.py 中一个常见的占位符，如果命令行没有提供，需要处理
    # # 这一部分不在 argparse 的职责范围内，但鉴于你的 config.py 大量使用，
    # # 可以在这里做一些后处理。
    # # 更好的做法是在实际使用这些路径时进行替换，或者让用户通过命令行指定完整路径。
    # # 这里为了演示，我们假设将其替换为当前用户桌面下的 AssertionForge 路径
    # base_desktop_path = Path.home() / 'Desktop' / 'AssertionForge'
    # for key, value in final_flags.__dict__.items():
    #     if isinstance(value, str) and '/<path>/<to>/' in value:
    #         # 简单替换示例，可能需要更复杂的逻辑来处理不同路径
    #         # 假设 /<path>/<to>/ 后面跟着的是相对于 AssertionForge 根目录的路径
    #         # 例如 /<path>/<to>/apb/apbi2c_spec.pdf 应该被替换为
    #         # C:\\Users\\huijie\\Desktop\\AssertionForge\\apb\\apbi2c_spec.pdf
    #         # 这个替换逻辑非常依赖于实际的路径结构，请根据实际情况调整。
    #         replaced_path = value.replace('/<path>/<to>/', str(base_desktop_path.as_posix() + '/')).replace('/', '\\')
    #         setattr(final_flags, key, replaced_path)
    #     elif isinstance(value, list):
    #         # 如果是路径列表，也需要遍历替换
    #         new_list = []
    #         for item in value:
    #             if isinstance(item, str) and '/<path>/<to>/' in item:
    #                 replaced_path = item.replace('/<path>/<to>/', str(base_desktop_path.as_posix() + '/')).replace('/', '\\')
    #                 new_list.append(replaced_path)
    #             else:
    #                 new_list.append(item)
    #         setattr(final_flags, key, new_list)


    return final_flags

# 如果你需要测试这个模块，可以在这里添加一些测试代码
if __name__ == '__main__':
    print("Running parse_command_line_args() for testing...")
    print("--------------------------------------------------")
    print("Initial FLAGS from config.py (before parsing args):")
    try:
        from config import FLAGS as initial_flags_for_test
        for k, v in initial_flags_for_test.__dict__.items():
            print(f"  {k}: {v}")
    except ImportError:
        print("  (config.py not found or import error for testing purposes)")
    print("--------------------------------------------------")


    # 模拟命令行参数，例如：
    # python command_line_args.py --task use_KG --design_name ethmac --DEBUG --llm_model gpt-4 --max_tokens_per_prompt 9000
    # python command_line_args.py --dynamic_prompt_settings '{"rag": {"enabled": false}}'
    print("\nAttempting to parse command line arguments...")
    test_flags = parse_command_line_args()
    print("\nParsed and Final FLAGS:")
    for k, v in test_flags.__dict__.items():
        print(f"  {k}: {v}")

    # 验证一些特定参数
    # print(f"\nTask: {test_flags.task}")
    # print(f"Design Name: {test_flags.design_name}")
    # print(f"Debug Mode: {test_flags.DEBUG}")
    # print(f"LLM Model: {test_flags.llm_model}")
    # print(f"Max Tokens Per Prompt: {test_flags.max_tokens_per_prompt}")
    # if hasattr(test_flags, 'dynamic_prompt_settings'):
    #     print(f"Dynamic Prompt Settings (rag enabled): {test_flags.dynamic_prompt_settings.get('rag', {}).get('enabled')}")