import subprocess
import re
import sys
import optuna

# 안드로이드 기기 내부의 작업 디렉토리를 지정하세요.
# (예: 실행 파일과 모델 파일들이 위치한 경로)
ANDROID_DIR = "/data/local/tmp/youngmin/eagle-2/benchmark"

# ADB에서 실행할 기본 커맨드 모음
# (실제 안드로이드 기기에 복사된 모델 이름과 경로에 맞게 맞춰주셔야 합니다.)
CMD_ARGS = [
    f"cd {ANDROID_DIR} &&",          # 작업 폴더로 이동 후 실행
    "/data/local/tmp/youngmin/eagle-2/benchmark/llama-speculative-eagle-2-search",   # 안드로이드용으로 크로스컴파일된 실행 파일 이름
    "-m", "/data/local/tmp/youngmin/models/vicuna_q4_0_output4.gguf",      # 기기 내 타겟 모델 경로 
    "-md", "/data/local/tmp/youngmin/models/EAGLE_q4_0_output4.gguf", # 기기 내 드래프트 모델 경로
    "-f", "/data/local/tmp/youngmin/datasets/prompt.txt",
    "-c", "0",
    "--top-p", "1.0",
    "--min-p", "0.0",
    "--temp", "0.0",
    "--draft-max", "500",
    "--draft-min", "1",
    "--n-predict", "100",            # 탐색을 위해 임시로 낮춘 생성 개수
    "-np", "500",
    "-s", "1234",
    "-kvu"
]

def run_experiment(n_depth, top_k, rerank_k):
    # 파라미터 추가
    current_args = CMD_ARGS.copy()
    current_args.extend(["--n-depth", str(n_depth)])
    current_args.extend(["--top-k", str(top_k)])
    
    if rerank_k > 0:
        current_args.extend(["--rerank-k", str(rerank_k)])
    else:
        current_args.extend(["--no-rerank"])

    # ADB shell에 던지기 위해 배열을 하나의 긴 문자열로 합침 
    # (주의: 큰따옴표 안의 경로나 공백이 있다면 적절한 이스케이핑이 필요할 수 있습니다.)
    full_android_cmd = " ".join(current_args)
    adb_cmd = ["adb", "shell", full_android_cmd]

    rk_str = "no-rerank" if rerank_k == 0 else str(rerank_k)
    print(f"ADB Testing >> n_depth={n_depth:<2} | top_k={top_k:<2} | rerank={rk_str:<9} => ", end="")
    sys.stdout.flush()

    try:
        # PC에서 'adb shell ...' 명령어를 실행시키고 그 출력을 받아옴
        result = subprocess.run(adb_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout + "\n" + result.stderr
        
        # 기기에서 나온 로그에서 정규표현식으로 성능 지표 파싱
        tps_match = re.search(r"decoded\s+\d+\s+tokens in\s+[\d.]+\s+seconds,\s+speed:\s+([\d.]+)\s+t/s", output)
        if not tps_match:
            print("Failed (Could not parse TPS from ADB stdout)")
            return None

        tps = float(tps_match.group(1))
        print(f"TPS: {tps:5.2f}")
        return tps

    except Exception as e:
        print(f"Error: {e}")
        return None

def objective(trial):
    # 하이퍼파라미터 제안
    n_depth = trial.suggest_int('n_depth', 3, 7)
    top_k = trial.suggest_int('top_k', 1, 15)
    
    use_rerank = trial.suggest_categorical('use_rerank', [True, False])
    if use_rerank:
        rerank_k = trial.suggest_int('rerank_k', 1, 59)
    else:
        rerank_k = 0

    max_tree_size_approx = sum([top_k ** i for i in range(1, n_depth + 1)])
    if use_rerank and rerank_k > max_tree_size_approx:
        print(f"Skipping >> rerank_k({rerank_k}) > max tree branches. Pruned.")
        raise optuna.exceptions.TrialPruned()

    tps = run_experiment(n_depth, top_k, rerank_k)
    
    if tps is None:
        raise optuna.exceptions.TrialPruned()
        
    return tps

def main():
    print("=" * 70)
    print("🚀 Starting Remote Bayesian Optimization via ADB & Optuna")
    print("   Optuna runs on Host PC, Executions run on connected Android Device")
    print("=" * 70)

    # Median pruning으로 너무 안 좋은 결과가 예상되면 스킵하는 기능도 켜줍니다.
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    
    # 총 40~50회 정도 진행해보는 것을 권장합니다.
    study.optimize(objective, n_trials=100)

    print("\n" + "=" * 70)
    print("🏆 Best Optuna Configuration Found for Android!")
    print("=" * 70)
    
    best_trial = study.best_trial
    print(f"  Best TPS: {best_trial.value:.2f} t/s")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")
        
    rerank_val = best_trial.params.get('rerank_k', 0)
    rerank_cmd = f"--rerank-k {rerank_val}" if best_trial.params.get('use_rerank') else "--no-rerank"
    print(f"\n  Final Ideal Command Params:")
    print(f"  --n-depth {best_trial.params['n_depth']} --top-k {best_trial.params['top_k']} {rerank_cmd}")
    print("=" * 70)

if __name__ == "__main__":
    main()
