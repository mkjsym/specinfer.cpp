import subprocess
import re
import itertools
import sys

# 테스트할 파라미터 후보군
N_DEPTH_LIST = [3, 4, 5, 6, 7]
TOP_K_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
RERANK_K_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]  # 0은 --no-rerank 를 의미합니다.

# 기본 실행 커맨드 (고정 파라미터)
BASE_CMD = [
    "build/bin/llama-speculative-eagle-2",
    "-m", "/data/youngmin/models/eagle_models/vicuna.gguf",
    "-md", "/data/youngmin/models/eagle_models/EAGLE-vicuna7b.gguf",
    "-f", "prompt.txt",
    "-c", "0",
    "--top-p", "1.0",
    "--min-p", "0.0",
    "--temp", "0.0",
    "--draft-max", "500",
    "--draft-min", "1",
    "--n-predict", "200",
    "-ngl", "40",
    "-ngld", "20",
    "-np", "500",
    "-s", "1234",
    "-kvu"
]

def run_experiment(n_depth, top_k, rerank_k):
    cmd = BASE_CMD.copy()
    cmd.extend(["--n-depth", str(n_depth)])
    cmd.extend(["--top-k", str(top_k)])
    
    if rerank_k > 0:
        cmd.extend(["--rerank-k", str(rerank_k)])
    else:
        cmd.extend(["--no-rerank"])
        
    print(f"n_depth={n_depth:<2} | top_k={top_k:<2} | rerank={'no-rerank' if rerank_k==0 else rerank_k:<9} => ", end="")
    sys.stdout.flush()

    try:
        # 실행 및 결과 캡처
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout + "\n" + result.stderr
        
        # 정규표현식으로 성능 지표 파싱
        # decoded  200 tokens in   16.737 seconds, speed:    11.950 t/s
        tps_match = re.search(r"decoded\s+\d+\s+tokens in\s+[\d.]+\s+seconds,\s+speed:\s+([\d.]+)\s+t/s", output)
        avg_len_match = re.search(r"Avg length:\s+([\d.]+)", output)
        draft_lat_match = re.search(r"avg drafting latency:\s+([\d.]+)\s*ms", output)
        verif_lat_match = re.search(r"avg verification latency:\s+([\d.]+)\s*ms", output)

        if not tps_match:
            print("Failed (Could not parse TPS from output)")
            return None

        tps = float(tps_match.group(1))
        avg_len = float(avg_len_match.group(1)) if avg_len_match else 0.0
        draft_lat = float(draft_lat_match.group(1)) if draft_lat_match else 0.0
        verif_lat = float(verif_lat_match.group(1)) if verif_lat_match else 0.0

        print(f"TPS: {tps:5.2f} | Avg Accept Len: {avg_len:4.2f} | Draft Lat: {draft_lat:5.2f}ms | Verif Lat: {verif_lat:5.2f}ms")
        
        return {
            "n_depth": n_depth,
            "top_k": top_k,
            "rerank_k": rerank_k,
            "tps": tps,
            "avg_len": avg_len,
            "draft_lat": draft_lat,
            "verif_lat": verif_lat
        }

    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    results = []
    
    combinations = list(itertools.product(N_DEPTH_LIST, TOP_K_LIST, RERANK_K_LIST))
    print("=" * 90)
    print(f"Starting Hyperparameter Search (Total {len(combinations)} combinations)")
    print("=" * 90)
    
    for i, (n_depth, top_k, rerank_k) in enumerate(combinations):
        print(f"[{i+1:>2}/{len(combinations)}] ", end="")
        res = run_experiment(n_depth, top_k, rerank_k)
        if res:
            results.append(res)
            
    if not results:
        print("\nNo successful runs were recorded.")
        return

    # TPS 기준으로 내림차순 정렬
    results.sort(key=lambda x: x["tps"], reverse=True)
    
    print("\n" + "=" * 90)
    print("🎯 Search Results (Sorted by TPS in descending order)")
    print("=" * 90)
    print(f"{'Rank':<5}| {'n_depth':<10}| {'top_k':<10}| {'rerank_k':<12}| {'TPS (t/s)':<12}| {'Avg Len':<10}| {'Draft Lat':<12}| {'Verif Lat':<12}")
    print("-" * 90)
    
    for i, r in enumerate(results):
        rk_str = "no-rerank" if r['rerank_k'] == 0 else str(r['rerank_k'])
        print(f"{i+1:<5}| {r['n_depth']:<10}| {r['top_k']:<10}| {rk_str:<12}| {r['tps']:<12.2f}| {r['avg_len']:<10.2f}| {r['draft_lat']:>7.2f} ms| {r['verif_lat']:>7.2f} ms")
    
    best = results[0]
    best_cmd_rk = f"--rerank-k {best['rerank_k']}" if best['rerank_k'] > 0 else "--no-rerank"
    
    print("\n" + "=" * 90)
    print("🏆 Best Configuration Found!")
    print(f"  Command Params: --n-depth {best['n_depth']} --top-k {best['top_k']} {best_cmd_rk}")
    print(f"  Performance:    TPS: {best['tps']:.2f} t/s / Avg Accept Length: {best['avg_len']:.2f} / Draft Latency: {best['draft_lat']:.2f} ms")
    print("=" * 90)

if __name__ == "__main__":
    main()
