#!/bin/bash
# ===========================================
# MAIA Workspace 환경 설정 스크립트
# 컨테이너 재시작 후 이 스크립트 실행: ./setup_env.sh
# ===========================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

echo "🔧 MAIA 환경 설정 시작..."
echo "📍 작업 디렉토리: $SCRIPT_DIR"

# 1. 기존 venv 정리 (있으면)
if [ -d "$VENV_DIR" ]; then
    echo "🗑️  기존 .venv 제거..."
    rm -rf "$VENV_DIR"
fi

# 기존 myenv도 정리
if [ -d "$SCRIPT_DIR/myenv" ]; then
    echo "🗑️  기존 myenv 제거..."
    rm -rf "$SCRIPT_DIR/myenv"
fi

# 2. venv 생성 (pip 없이 먼저 생성 후 get-pip.py로 설치)
echo "🐍 Python venv 생성 중..."
python3 -m venv "$VENV_DIR" --without-pip 2>/dev/null || {
    echo "⚠️  --without-pip으로 시도..."
    python3 -m venv "$VENV_DIR" --without-pip
}

# 3. pip 수동 설치
echo "📦 pip 설치 중..."
source "$VENV_DIR/bin/activate"
curl -sS https://bootstrap.pypa.io/get-pip.py | python3

# 4. pip 업그레이드
echo "⬆️  pip 업그레이드..."
pip install --upgrade pip

# 5. 프로젝트별 requirements 설치
echo "📚 tamper-resistance 패키지 설치 중..."
if [ -f "$SCRIPT_DIR/tamper-resistance/requirements.txt" ]; then
    pip install -r "$SCRIPT_DIR/tamper-resistance/requirements.txt"
fi

# 6. 추가 공통 패키지 (필요시 여기에 추가)
echo "🔧 추가 유틸리티 설치..."
pip install ipython jupyter black isort

echo ""
echo "✅ 환경 설정 완료!"
echo ""
echo "📌 사용법:"
echo "   source $VENV_DIR/bin/activate"
echo ""
echo "📌 또는 간단히:"
echo "   source .venv/bin/activate"
echo ""
