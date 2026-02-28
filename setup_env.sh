#!/bin/bash
# ===========================================
# MAIA Workspace 환경 설정 스크립트
# 
# 사용법:
#   ./setup_env.sh              # 메뉴 선택
#   ./setup_env.sh tamper       # tamper-resistance 환경
#   ./setup_env.sh honmun       # Honmun 환경
#   ./setup_env.sh all          # 모든 프로젝트 환경 (별도 venv)
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================
# 공통 함수
# ============================================
create_venv() {
    local venv_path=$1
    echo -e "${BLUE}🐍 Python venv 생성 중: $venv_path${NC}"
    
    # 기존 venv 제거
    if [ -d "$venv_path" ]; then
        echo -e "${YELLOW}🗑️  기존 venv 제거...${NC}"
        rm -rf "$venv_path"
    fi
    
    # venv 생성
    python3 -m venv "$venv_path" --without-pip 2>/dev/null || \
        python3 -m venv "$venv_path" --without-pip
    
    # pip 설치
    source "$venv_path/bin/activate"
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    pip install --upgrade pip
}

install_common_tools() {
    echo -e "${BLUE}🔧 공통 유틸리티 설치...${NC}"
    pip install ipython jupyter black isort
}

# ============================================
# 프로젝트별 설치 함수
# ============================================
setup_tamper_resistance() {
    local venv_path="$SCRIPT_DIR/.venv-tamper"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📦 tamper-resistance 환경 설정${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    create_venv "$venv_path"
    
    if [ -f "$SCRIPT_DIR/tamper-resistance/requirements.txt" ]; then
        echo -e "${BLUE}📚 tamper-resistance 패키지 설치...${NC}"
        pip install -r "$SCRIPT_DIR/tamper-resistance/requirements.txt"
    fi
    
    install_common_tools
    
    echo ""
    echo -e "${GREEN}✅ tamper-resistance 환경 완료!${NC}"
    echo -e "   활성화: ${YELLOW}source .venv-tamper/bin/activate${NC}"
}

setup_honmun() {
    local venv_path="$SCRIPT_DIR/.venv-honmun"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📦 Honmun 환경 설정${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    create_venv "$venv_path"
    
    # PyTorch 2.6.0 + CUDA 12.4
    echo -e "${BLUE}🔥 PyTorch 2.6.0 (CUDA 12.4) 설치...${NC}"
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    
    # Honmun 설치
    if [ -d "$SCRIPT_DIR/Honmun" ]; then
        echo -e "${BLUE}📚 Honmun 패키지 설치 (editable)...${NC}"
        pip install -e "$SCRIPT_DIR/Honmun"
        
        # git submodule 업데이트
        echo -e "${BLUE}🔄 Git submodule 업데이트...${NC}"
        cd "$SCRIPT_DIR/Honmun"
        git submodule update --init --recursive
        cd "$SCRIPT_DIR"
    fi
    
    install_common_tools
    
    echo ""
    echo -e "${GREEN}✅ Honmun 환경 완료!${NC}"
    echo -e "   활성화: ${YELLOW}source .venv-honmun/bin/activate${NC}"
}

# ============================================
# 메뉴
# ============================================
show_menu() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}   MAIA Workspace 환경 설정${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "  1) tamper-resistance  (torch==2.4)"
    echo "  2) Honmun             (torch==2.6.0+cu124)"
    echo "  3) 모두 설치          (별도 venv로 분리)"
    echo "  q) 종료"
    echo ""
    echo -n "선택: "
}

# ============================================
# 메인
# ============================================
main() {
    case "${1:-}" in
        tamper|1)
            setup_tamper_resistance
            ;;
        honmun|2)
            setup_honmun
            ;;
        all|3)
            setup_tamper_resistance
            deactivate 2>/dev/null || true
            setup_honmun
            ;;
        "")
            show_menu
            read choice
            case $choice in
                1) setup_tamper_resistance ;;
                2) setup_honmun ;;
                3) 
                    setup_tamper_resistance
                    deactivate 2>/dev/null || true
                    setup_honmun
                    ;;
                q|Q) echo "종료합니다." ;;
                *) echo -e "${RED}잘못된 선택입니다.${NC}" ;;
            esac
            ;;
        *)
            echo "사용법: $0 [tamper|honmun|all]"
            exit 1
            ;;
    esac
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}📌 환경 활성화 명령어:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  tamper-resistance: ${YELLOW}source .venv-tamper/bin/activate${NC}"
    echo -e "  Honmun:            ${YELLOW}source .venv-honmun/bin/activate${NC}"
    echo ""
}

main "$@"
