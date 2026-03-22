import subprocess
import re
import sys
import optuna

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
    "--n-predict", "100",  # 탐색용을 위해 100을 우선 추천드립니다. 최적값을 찾은 후 500으로 재검증하세요.
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

    # 실행 정보 출력
    rk_str = "no-rerank" if rerank_k == 0 else str(rerank_k)
    print(f"Testing >> n_depth={n_depth:<2} | top_k={top_k:<2} | rerank={rk_str:<9} => ", end="")
    sys.stdout.flush()

    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout + "\n" + result.stderr
        
        tps_match = re.search(r"decoded\s+\d+\s+tokens in\s+[\d.]+\s+seconds,\s+speed:\s+([\d.]+)\s+t/s", output)
        if not tps_match:
            print("Failed (Could not parse TPS)")
            return None

        tps = float(tps_match.group(1))
        print(f"TPS: {tps:5.2f}")
        return tps

    except Exception as e:
        print(f"Error: {e}")
        return None

def objective(trial):
    # 1. 하이퍼파라미터 제안
    n_depth = trial.suggest_int('n_depth', 3, 7)
    top_k = trial.suggest_int('top_k', 1, 15)
    
    # Reranking 사용 여부 결정 (0은 사용 안 함, 1~59는 사용함)
    use_rerank = trial.suggest_categorical('use_rerank', [True, False])
    if use_rerank:
        rerank_k = trial.suggest_int('rerank_k', 1, 59)
    else:
        rerank_k = 0

    # 2. 의미 없는 검색 공간 가지치기 (Pruning)
    # n_depth와 top_k를 고려한 대략적인 최대 트리의 크기보다 큰 rerank_k는 의미가 없으므로 Prune 시킵니다.
    max_tree_size_approx = sum([top_k ** i for i in range(1, n_depth + 1)])
    if use_rerank and rerank_k > max_tree_size_approx:
        print(f"Skipping >> rerank_k({rerank_k}) is larger than max tree branches. Pruned.")
        raise optuna.exceptions.TrialPruned()

    # 3. 모델 돌리기 & 평가
    tps = run_experiment(n_depth, top_k, rerank_k)
    
    if tps is None:
        raise optuna.exceptions.TrialPruned()
        
    return tps

def main():
    print("=" * 70)
    print("🚀 Starting Bayesian Optimization via Optuna")
    print("=" * 70)

    # Optuna 스터디 생성 (tps를 '최대화(maximize)' 하는 방향으로)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # 예: 50회의 스마트 탐색 (Random Search보다 50회 내에 최적해를 훨씬 빠르게 찾음)
    study.optimize(objective, n_trials=200)

    print("\n" + "=" * 70)
    print("🏆 Best Optuna Configuration Found!")
    print("=" * 70)
    
    best_trial = study.best_trial
    print(f"  Best TPS: {best_trial.value:.2f} t/s")
    print(f"  Best Params: ")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")
        
    rerank_val = best_trial.params.get('rerank_k', 0)
    rerank_cmd = f"--rerank-k {rerank_val}" if best_trial.params.get('use_rerank') else "--no-rerank"
    print(f"\n  Run Command:")
    print(f"  --n-depth {best_trial.params['n_depth']} --top-k {best_trial.params['top_k']} {rerank_cmd}")
    print("=" * 70)

if __name__ == "__main__":
    main()
