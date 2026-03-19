import re

log_path = '/home/youngmin/workspace/New_specinfer.cpp/specinfer.cpp/benchmark/logs/EAGLE-1-6-mobile_Q4_0_log_2026-03-18_00-51-06.txt'

with open(log_path, 'r') as f:
    lines = f.readlines()

test_cases = []
current_case = None
case_data = {}

question_pattern = re.compile(r'>>> Question ID:\s+(\d+)')
avg_length_pattern = re.compile(r'Avg length:\s*([\d\.]+)')
drafting_latency_pattern = re.compile(r'avg (?:drafting|decoding) latency:\s*([\d\.]+)\s*ms')
verification_latency_pattern = re.compile(r'avg verification latency:\s*([\d\.]+)\s*ms')
avg_td_pattern = re.compile(r'avg T_d:\s*([\d\.]+)\s*ms')
overall_tps_pattern = re.compile(r'decoded\s+(\d+)\s+tokens in\s+([\d\.]+)\s+seconds,\s+speed:\s+([\d\.]+)\s+t/s')

for line in lines:
    q_match = question_pattern.search(line)
    if q_match:
        if current_case is not None and len(case_data) >= 3:
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
        
    m_overall_tps = overall_tps_pattern.search(line)
    if m_overall_tps:
        case_data['overall_tps'] = float(m_overall_tps.group(3))

# Add the last case
if current_case is not None and len(case_data) >= 3:
    case_data['id'] = current_case
    test_cases.append(case_data)

for tc in test_cases:
    print(f"Question ID {tc['id']}: Verification = {tc.get('verification_latency', 0):.2f} ms, "
          f"Draft Decode (T_d) = {tc.get('draft_decode_latency', 0):.2f} ms, "
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
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Avg Verification Latency (Target): {avg_verif:.2f} ms")
    print(f"Avg Decode Latency       (Draft): {avg_draft_dec:.2f} ms")
    print(f"Avg Drafting Step Latency:        {avg_drafting_step:.2f} ms")
    print(f"Average Acceptance Length:        {avg_acc_len:.2f}")
    print(f"Overall Average TPS:              {avg_overall_tps:.2f} tokens/s")
else:
    print("No test cases found in log.")
