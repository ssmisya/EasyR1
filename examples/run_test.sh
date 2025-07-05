#!/bin/bash

# 定义颜色
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}===== 测试脚本 =====${NC}"

# 检查是否有代理环境变量
if [ -n "$http_proxy" ] || [ -n "$https_proxy" ]; then
    echo -e "${GREEN}检测到代理环境变量:${NC}"
    echo "http_proxy=$http_proxy"
    echo "https_proxy=$https_proxy"
else
    echo -e "${RED}未检测到代理环境变量${NC}"
fi

echo ""
echo -e "${YELLOW}请选择运行模式:${NC}"
echo "1) 本地服务器不使用代理，wandb使用代理（推荐）"
echo "2) 全部不使用代理"
echo "3) 全部使用代理"
echo "4) 自定义服务器URL"
echo ""

read -p "请输入选项 [1-4]: " option

case $option in
    1)
        echo -e "${GREEN}运行模式: 本地服务器不使用代理，wandb使用代理${NC}"
        python test.py --use-proxy-for-wandb
        ;;
    2)
        echo -e "${GREEN}运行模式: 全部不使用代理${NC}"
        # 临时清除代理环境变量
        export _OLD_HTTP_PROXY=$http_proxy
        export _OLD_HTTPS_PROXY=$https_proxy
        unset http_proxy
        unset https_proxy
        python test.py
        # 恢复代理环境变量
        export http_proxy=$_OLD_HTTP_PROXY
        export https_proxy=$_OLD_HTTPS_PROXY
        ;;
    3)
        echo -e "${GREEN}运行模式: 全部使用代理${NC}"
        python test.py --use-proxy-for-wandb
        ;;
    4)
        read -p "请输入服务器URL: " server_url
        echo -e "${GREEN}运行模式: 使用自定义URL ${server_url}${NC}"
        read -p "是否为wandb使用代理? (y/n): " use_proxy
        if [[ $use_proxy == "y" ]]; then
            python test.py --use-proxy-for-wandb --server-url "$server_url"
        else
            # 临时清除代理环境变量
            export _OLD_HTTP_PROXY=$http_proxy
            export _OLD_HTTPS_PROXY=$https_proxy
            unset http_proxy
            unset https_proxy
            python test.py --server-url "$server_url"
            # 恢复代理环境变量
            export http_proxy=$_OLD_HTTP_PROXY
            export https_proxy=$_OLD_HTTPS_PROXY
        fi
        ;;
    *)
        echo -e "${RED}无效选项，退出${NC}"
        exit 1
        ;;
esac 