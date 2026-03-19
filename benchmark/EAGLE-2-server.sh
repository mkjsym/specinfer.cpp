#!/bin/bash

# 스크립트: JSONL 파일을 읽어 각 질문의 첫 번째 턴에 템플릿을 적용하여 llama-speculative-eagle 추론을 수행합니다.
#          '#'로 시작하는 주석 처리된 줄은 건너뜁니다.
#          모든 실행 결과를 로그 파일로 저장합니다.
# 사용법: ./run_inference_with_log.sh <입력_JSONL_파일>
# 예시:   ./run_inference_with_log.sh your_data.jsonl

# --- 설정 ---
# 실행 파일 경로
EXECUTABLE="build/bin/llama-speculative-eagle-2"
# 모델 및 추론 파라미터
MODEL_PATH_TGT="/data/youngmin/models/eagle_models/vicuna_q4_0_output4.gguf"
MODEL_PATH_DFT="/data/youngmin/models/eagle_models/EAGLE-vicuna7b_q4_0_output4.gguf"
INFERENCE_ARGS="-c 0 --color --top-k 10 -fa off --top-p 1.0 --min-p 0.0 --temp 0.0 --draft-max 500 --draft-min 1 --n-predict 200 -ngl 40 -ngld 20 -np 200 -s 1234 -kvu"
# [추가] 로그 파일 이름 설정 (날짜와 시간 포함)
LOG_FILE="./benchmark/logs/EAGLE-2-server_Q4_0_$(date +%Y-%m-%d_%H-%M-%S).txt"

# --- 스크립트 시작 ---

# [수정] 전체 스크립트의 출력을 로그 파일과 콘솔에 동시에 보냅니다.
{
    # 1. 필수 유틸리티 (jq) 설치 여부 확인
    if ! command -v jq &> /dev/null
    then
        echo "오류: 'jq'가 설치되어 있지 않습니다. 스크립트를 실행하려면 jq를 설치해주세요."
        exit 1
    fi

    # 2. 입력 파일 인자 확인
    if [ "$#" -ne 1 ]; then
        echo "사용법: $0 <입력_JSONL_파일>"
        exit 1
    fi

    INPUT_FILE="$1"

    # 3. 입력 파일 존재 여부 확인
    if [ ! -f "$INPUT_FILE" ]; then
        echo "오류: 입력 파일 '$INPUT_FILE'을(를) 찾을 수 없습니다."
        exit 1
    fi
    
    echo "--- 추론을 시작하며, 모든 결과는 '$LOG_FILE' 파일에 저장됩니다. ---"

    # 4. JSONL 파일을 한 줄씩 읽어 추론 실행
    while IFS= read -r line
    do
        # 빈 줄은 건너뛰기
        if [ -z "$line" ]; then
            continue
        fi

        # '#'으로 시작하는 주석 줄은 건너뛰기
        if [[ "$line" == \#* ]]; then
            echo "주석 처리된 줄을 건너뜁니다: $line"
            continue
        fi

        # jq를 사용하여 question_id와 첫 번째 turn의 내용을 추출
        question_id=$(echo "$line" | jq -r '.question_id')
        original_prompt=$(echo "$line" | jq -r '.turns[0]')

        # 지정된 템플릿을 프롬프트에 적용합니다.
        TEMPLATE="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: %s ASSISTANT:"
        formatted_prompt=$(printf "$TEMPLATE" "$original_prompt")


        echo "============================================================"
        echo ">>> Question ID: $question_id 추론을 시작합니다..."
        echo "============================================================"

        # 명령어 실행 (수정된 formatted_prompt 사용)
        "$EXECUTABLE" \
            -m "$MODEL_PATH_TGT" \
            -md "$MODEL_PATH_DFT" \
            -p "$formatted_prompt" \
            $INFERENCE_ARGS

        echo # 각 추론 결과 사이에 공백 줄 추가

    done < "$INPUT_FILE"

    echo "--- 모든 추론이 완료되었습니다. ---"

} 2>&1 | tee "$LOG_FILE"
