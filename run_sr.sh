#!/bin/bash

# 设置环境变量
export PYTHONPATH="$PYTHONPATH:$(pwd)"

# 默认参数
ENABLE_LLM=true
CONFIG_PATH="config/default_config.yaml"
DEFAULT_CONFIG_PATH="config/default_config.yaml"

API_KEY=${SR_API_KEY:-""}  # 从环境变量获取 API_KEY，如果未设置则为空

# 帮助函数
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --help                 显示帮助信息"
    echo "  --no-llm                   禁用 LLM 功能"
    echo "  -c, --config <path>        指定配置文件路径"
    echo "  -k, --api-key <key>        设置 API KEY (也可通过 SR_API_KEY 环境变量设置)"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        --no-llm)
            ENABLE_LLM=false
            shift
            ;;
        -c|--config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 检查 LLM 功能是否启用且 API_KEY 是否设置
if [ "$ENABLE_LLM" = true ] && [ -z "$API_KEY" ]; then
    echo "错误: 启用 LLM 功能时需要设置 API KEY"
    echo "可以通过以下方式设置:"
    echo "1. 环境变量: export SR_API_KEY=your_api_key"
    echo "2. 命令行参数: -k your_api_key"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "未指定配置文件，将采用默认的配置文件: $DEFAULT_CONFIG_PATH"
    CONFIG_PATH="$DEFAULT_CONFIG_PATH"
fi

# 创建输出目录
mkdir -p output

# 如果通过命令行提供了 API_KEY，则设置环境变量
if [ -n "$API_KEY" ]; then
    export SR_API_KEY="$API_KEY"
fi

# 运行 Python 脚本
echo "启动符号回归生成..."
echo "配置文件: $CONFIG_PATH"
echo "LLM 功能: $ENABLE_LLM"

python src/run_sr.py \
    --enable_llm=$ENABLE_LLM \
    --config_path="$CONFIG_PATH"

# 检查运行结果
if [ $? -eq 0 ]; then
    echo "运行完成"
else
    echo "运行失败"
    exit 1
fi 