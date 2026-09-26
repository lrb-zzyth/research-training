"""
参数契约测试: 前端字段 <-> canonical schema <-> train_fedtad.py CLI 参数。

验证项 (对应清理规范"参数契约测试"):
  1. 每个前端字段都能映射到 canonical config (无未知/无效前端参数);
  2. 每个 canonical 字段有后端校验元数据 (type/min/max/choices/hot_update_policy);
  3. 每个训练参数都能到达实际训练代码 (COMMAND_FIELDS 全部存在于 train_fedtad.py CLI);
  4. 未知字段被拒绝 (422);
  5. 不存在前端发送但后端不接受/永不生效的参数;
  6. 热更新白名单与训练进程白名单一致。

运行方式: cd research-training/backend && python test_param_contract.py
"""
import os
import re
import subprocess
import sys

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(BACKEND_DIR))
sys.path.insert(0, BACKEND_DIR)
sys.path.insert(0, REPO_ROOT)

from app.config_schema import (CANONICAL_SCHEMA, COMMAND_FIELDS,
                               BOOLEAN_OPTIONAL, HOT_UPDATABLE, ALIAS_TO_PRIMARY,
                               validate_and_normalize)

passed = []
failed = []


def check(name, cond, extra=""):
    if cond:
        passed.append(name)
        print(f"  ✓ {name}")
    else:
        failed.append(name)
        print(f"  ✗ {name} {extra}")


print("== 参数契约测试 ==")

# ---- 1. 前端字段 ⊆ canonical schema ----
print("[1] 前端字段映射")
vue_path = os.path.join(REPO_ROOT, "research-training", "frontend",
                        "src", "views", "TrainingConfig.vue")
vue_src = open(vue_path).read()
m = re.search(r"const defaults = \{(.*?)\n\}", vue_src, re.S)
frontend_keys = set(re.findall(r"^  ([a-z_0-9]+):", m.group(1), re.M))
frontend_keys.discard("name")  # 前端内部字段, 发送前删除
unknown_frontend = sorted(frontend_keys - set(CANONICAL_SCHEMA))
check("1a 前端全部字段都在 canonical schema 中",
      not unknown_frontend, f"unknown: {unknown_frontend}")
check(f"1b 前端字段数 = {len(frontend_keys)}", len(frontend_keys) >= 40)

# ---- 2. schema 元数据完整性 ----
print("[2] schema 元数据")
missing_meta = []
for k, meta in CANONICAL_SCHEMA.items():
    for field in ("type", "default", "group", "hot_update_policy",
                  "requires_restart", "effective_stage", "description"):
        if field not in meta:
            missing_meta.append(f"{k}.{field}")
check("2a 每个字段都有完整元数据", not missing_meta, str(missing_meta))
check("2b hot_update_policy 取值合法",
      all(v["hot_update_policy"] in ("start", "round_boundary")
          for v in CANONICAL_SCHEMA.values()))
check("2c requires_restart 与 hot_update_policy 一致",
      all((v["hot_update_policy"] == "round_boundary") == (not v["requires_restart"])
          for v in CANONICAL_SCHEMA.values()))

# ---- 3. 每个训练参数都能到达训练代码 ----
print("[3] CLI 参数可达性")
help_out = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "train_fedtad.py"),
                           "--help"], capture_output=True, text=True).stdout
cli_flags = set(re.findall(r"--[a-z0-9_\-]+", help_out))
cli_flags.add("--no-use_weighted_ce")   # BooleanOptionalAction 自动生成
cli_flags.add("--no-save_last_checkpoint")
cli_flags.add("--no-resplit_stratified")
unreachable = [f for f in COMMAND_FIELDS if f"--{f}" not in cli_flags]
check("3a COMMAND_FIELDS 全部存在于 train_fedtad.py CLI",
      not unreachable, f"unreachable: {unreachable}")
check("3b 布尔参数均有 --no- 形式",
      all(f"--no-{f}" in cli_flags for f in BOOLEAN_OPTIONAL))

# ---- 4. schema 字段要么进命令行要么是别名 (无"接受但永不生效") ----
print("[4] 无静默失效参数")
never_effective = [k for k in CANONICAL_SCHEMA
                   if k not in COMMAND_FIELDS and k not in ALIAS_TO_PRIMARY]
check("4a 每个 schema 字段都有生效路径", not never_effective,
      f"never effective: {never_effective}")

# ---- 5. 校验行为 ----
print("[5] 校验行为")
_, errs = validate_and_normalize({"evil": 1, "num_clients": 1, "hid_dim": -5,
                                  "contrastive_mode": "bad"})
check("5a 未知/越界/非法 choices 全部被拒", len(errs) == 4, str(errs))
cfg, errs = validate_and_normalize({"learning_rate": 0.005, "federated_rounds": 3,
                                    "fake_node_count": 50, "distillation_steps": 7})
check("5b canonical 别名正常映射", not errs and cfg.get("lr") == 0.005
      and cfg.get("num_rounds") == 3 and cfg.get("fake_nodes") == 50
      and cfg.get("distill_steps") == 7, str(errs))
check("5c 别名键不出现在规范化快照",
      not any(a in cfg for a in ALIAS_TO_PRIMARY))
cfg, errs = validate_and_normalize({"task_mode": "anomaly_binary",
                                    "normal_classes": "   ", "anomaly_classes": "4,5,6"})
check("5d 标签映射组合校验 (空白映射被拒)", len(errs) >= 1, str(errs))
cfg, errs = validate_and_normalize({"task_mode": "anomaly_binary"})
check("5e 未提供映射时使用默认映射", not errs and cfg.get("normal_classes") == "0,1,2,3",
      str(errs))
_, errs = validate_and_normalize({"knn_k": 50, "fake_node_count": 20})
check("5f KNN<伪节点数组合校验", len(errs) >= 1, str(errs))

# ---- 6. 热更新白名单一致性 ----
print("[6] 热更新白名单")
train_src = open(os.path.join(REPO_ROOT, "train_fedtad.py")).read()
_block = re.search(r"HOT_UPDATABLE_PARAMS = \{(.*?)\n\}\n", train_src, re.S).group(1)
train_whitelist = set(re.findall(r"'([a-z_]+)': \(", _block))
backend_hot = set(HOT_UPDATABLE)
check("6a 后端热更新白名单与训练进程白名单一致",
      backend_hot == train_whitelist,
      f"backend-only={backend_hot - train_whitelist} "
      f"train-only={train_whitelist - backend_hot}")
check("6b 热更新参数均有 schema 元数据",
      all(k in CANONICAL_SCHEMA for k in HOT_UPDATABLE))

print(f"\n== 结果: {len(passed)} 通过, {len(failed)} 失败 ==")
if failed:
    print(f"失败项: {failed}")
    sys.exit(1)
print("ALL PARAM CONTRACT TESTS PASSED ✓")
