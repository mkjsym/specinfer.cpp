import re

log_path = '/home/youngmin/workspace/New_specinfer.cpp/specinfer.cpp/benchmark/logs/AR_mobile_Q4_0_2026-03-16_21-38-14.txt'

with open(log_path, 'r') as f:
    lines = f.readlines()

test_cases = []
current_case = None

question_pattern = re.compile(r'>>> Question ID: (\d+)')
eval_time_pattern = re.compile(r'eval time =.*?\(.*?([\d\.]+)\s*ms per token,\s*([\d\.]+)\s*tokens per second\)')

for line in lines:
    q_match = question_pattern.search(line)
    if q_match:
        current_case = q_match.group(1)
        continue
    
    # We specifically want 'eval time =' but not 'prompt eval time'
    if 'eval time =' in line and 'prompt' not in line:
        e_match = eval_time_pattern.search(line)
        if e_match and current_case is not None:
            latency = float(e_match.group(1))
            tps = float(e_match.group(2))
            test_cases.append({'id': current_case, 'latency': latency, 'tps': tps})
            current_case = None

for tc in test_cases:
    print(f"Question ID {tc['id']}: Decode Latency = {tc['latency']:.2f} ms/token, TPS = {tc['tps']:.2f}")

if len(test_cases) > 0:
    avg_latency = sum(tc['latency'] for tc in test_cases) / len(test_cases)
    avg_tps = sum(tc['tps'] for tc in test_cases) / len(test_cases)
    print("-" * 40)
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Average Decode Latency: {avg_latency:.2f} ms/token")
    print(f"Average TPS: {avg_tps:.2f}")
else:
    print("No test cases found.")
