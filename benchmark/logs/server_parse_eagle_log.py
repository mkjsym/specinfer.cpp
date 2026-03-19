import re

log_path = '/home/youngmin/workspace/New_specinfer.cpp/specinfer.cpp/benchmark/logs/EAGLE-2-server_2026-03-16_20-35-07.txt'

with open(log_path, 'r') as f:
    lines = f.readlines()

test_cases = []
current_case = None
case_data = {}

question_pattern = re.compile(r'>>> Question ID: (\d+)')
avg_length_pattern = re.compile(r'Avg length:\s*([\d\.]+)')
drafting_latency_pattern = re.compile(r'avg drafting latency:\s*([\d\.]+)\s*ms')
verification_latency_pattern = re.compile(r'avg verification latency:\s*([\d\.]+)\s*ms')
avg_td_pattern = re.compile(r'avg T_d:\s*([\d\.]+)\s*ms')

# Draft(llama_perf)와 Target(common_perf)의 TPS 추출을 위한 정규식
llama_perf_prompt_pattern = re.compile(r'llama_perf_context_print: prompt eval time =.*?([\d\.]+)\s*tokens per second\)')
common_perf_prompt_pattern = re.compile(r'common_perf_print: prompt eval time =.*?([\d\.]+)\s*tokens per second\)')

# Overall TPS extraction
overall_tps_pattern = re.compile(r'decoded\s+(\d+)\s+tokens in\s+([\d\.]+)\s+seconds,\s+speed:\s+([\d\.]+)\s+t/s')

for line in lines:
    q_match = question_pattern.search(line)
    if q_match:
        if current_case is not None and len(case_data) >= 4:
            case_data['id'] = current_case
            test_cases.append(case_data)
        current_case = q_match.group(1)
        case_data = {}
        continue
    
    if current_case is None:
        continue

    m_len = avg_length_pattern.search(line)
    if m_len:
        case_data['avg_length'] = float(m_len.group(1))
        
    m_draft = drafting_latency_pattern.search(line)
    if m_draft:
        case_data['drafting_latency'] = float(m_draft.group(1))

    m_verif = verification_latency_pattern.search(line)
    if m_verif:
        case_data['verification_latency'] = float(m_verif.group(1))
        
    m_td = avg_td_pattern.search(line)
    if m_td:
        case_data['draft_decode_latency'] = float(m_td.group(1))

    m_llama_tps = llama_perf_prompt_pattern.search(line)
    if m_llama_tps:
        case_data['draft_tps'] = float(m_llama_tps.group(1))
        
    m_common_tps = common_perf_prompt_pattern.search(line)
    if m_common_tps:
        case_data['target_tps'] = float(m_common_tps.group(1))
        
    m_overall_tps = overall_tps_pattern.search(line)
    if m_overall_tps:
        case_data['overall_tps'] = float(m_overall_tps.group(3))

# 마지막 케이스 추가
if current_case is not None and len(case_data) >= 4:
    case_data['id'] = current_case
    test_cases.append(case_data)

for tc in test_cases:
    print(f"Question ID {tc['id']}: Verification = {tc.get('verification_latency', 0):.2f} ms, "
          f"Draft Decode = {tc.get('draft_decode_latency', 0):.2f} ms, "
          f"Drafting Step = {tc.get('drafting_latency', 0):.2f} ms, "
          f"Acceptance Length = {tc.get('avg_length', 0):.2f}, "
          f"Overall TPS = {tc.get('overall_tps', 0):.2f}")

if len(test_cases) > 0:
    avg_verif = sum(tc.get('verification_latency', 0) for tc in test_cases) / len(test_cases)
    avg_draft_dec = sum(tc.get('draft_decode_latency', 0) for tc in test_cases) / len(test_cases)
    avg_drafting_step = sum(tc.get('drafting_latency', 0) for tc in test_cases) / len(test_cases)
    avg_acc_len = sum(tc.get('avg_length', 0) for tc in test_cases) / len(test_cases)
    avg_overall_tps = sum(tc.get('overall_tps', 0) for tc in test_cases) / len(test_cases)
    
    print("-" * 60)
    print(f"총 테스트 케이스: {len(test_cases)}개")
    print(f"타겟 모델 평균 Verification Latency: {avg_verif:.2f} ms")
    print(f"드래프트 모델 평균 Decode Latency (T_d): {avg_draft_dec:.2f} ms")
    print(f"평균 Drafting Step Latency:        {avg_drafting_step:.2f} ms")
    print(f"평균 Acceptance Length:              {avg_acc_len:.2f}")
    print(f"전체 평균 알고리즘 TPS (Overall TPS): {avg_overall_tps:.2f} tokens/s")
else:
    print("로그에서 테스트 케이스 결과를 찾지 못했습니다.")
