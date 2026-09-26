"""
后端平台 smoke test (真实服务, urllib, 无需额外依赖)

验证项 (对应清理规范"后端测试至少验证"):
  1. 服务可以启动 (health check)
  2. 核心 API 能注册 (auth/experiments/training 路由)
  3. 请求参数可以转换为算法配置 (start 响应中 command 含合法 CLI flag)
  4. 非法参数返回明确错误 (非 dict parameters -> 422 + detail)
  5. 训练任务可以创建 (start -> 201/200 + id)
  6. 任务状态可以查询 (status 端点 + 实验详情)
  7. 训练异常可以被平台读取 (失败数据集 -> status=failed, logs 含错误)
  8. 结果和日志接口可用 (logs/metrics 端点)
  9. 前端依赖的响应字段仍然存在 (ExperimentResponse 全部字段)

运行方式:
  cd research-training/backend && python test_backend_smoke.py
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 8011
BASE = f"http://127.0.0.1:{PORT}/api"


def http(method, path, body=None, token=None, timeout=30):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
        except Exception:
            detail = None
        return e.code, detail


def wait_health(timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            st, body = http("GET", "/health", timeout=3)
            if st == 200 and body.get("status") == "ok":
                return True
        except Exception:
            time.sleep(1)
    return False


def main():
    passed = []
    failed = []

    def check(name, cond, extra=""):
        if cond:
            passed.append(name)
            print(f"  ✓ {name}")
        else:
            failed.append(name)
            print(f"  ✗ {name} {extra}")

    print("== 后端 smoke test ==")

    # ---- 启动服务 ----
    print("[1] 启动 uvicorn 服务")
    env = dict(os.environ)
    env["PYTHONPATH"] = BACKEND_DIR
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=BACKEND_DIR, env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        check("1a 服务可以启动 (health check)", wait_health())

        # ---- 鉴权 ----
        print("[2] 鉴权")
        uname = f"smoke_{int(time.time())}"
        st, body = http("POST", "/auth/register",
                        {"username": uname, "password": "smokepass123",
                         "confirm_password": "smokepass123"})
        check("2a 注册接口可用", st == 200 and body.get("username") == uname)
        st, body = http("POST", "/auth/login",
                        {"username": uname, "password": "smokepass123"})
        token = body.get("access_token") if st == 200 else None
        check("2b 登录接口可用", st == 200 and token, str(body))
        st, body = http("GET", "/auth/me", token=token)
        check("2c 当前用户接口可用", st == 200 and body.get("username") == uname)
        st, body = http("GET", "/auth/me")
        check("2d 未授权访问返回 401/403", st in (401, 403))

        # ---- 非法参数 -> 明确错误 ----
        print("[3] 非法参数校验")
        st, body = http("POST", "/training/start",
                        {"name": "bad", "parameters": "not-a-dict"},
                        token=token)
        check("3a 非 dict parameters 返回 422 + detail",
              st == 422 and body is not None and "detail" in body, str(st))

        # ---- 参数转换与任务创建 ----
        print("[4] 参数转换 + 任务创建")
        params = {
            "root": "./dataset", "dataset": "Cora",
            "task_mode": "anomaly_binary",
            "normal_classes": "0,1,2,3", "anomaly_classes": "4,5,6",
            "num_clients": 2, "federated_rounds": 1, "num_epochs": 1,
            "learning_rate": 0.01, "hid_dim": 32, "dropout": 0.2,
            "use_weighted_ce": False,
            "contrastive_mode": "none", "distill_weighting": "none",
            "fake_node_count": 32, "knn_k": 5, "generator_steps": 1,
            "diffusion_steps": 4, "diffusion_hidden": 32,
            "rwr_subgraph_size": 3, "contrastive_batch_size": 8,
            "edge_perturb_ratio": 0.2, "seed": 2024,
        }
        st, exp = http("POST", "/training/start",
                       {"name": "smoke-valid-test", "parameters": params},
                       token=token)
        check("4a 训练任务可以创建 (start 200)", st == 200, str(st))
        if st == 200:
            # 响应字段完整性 (含平台集成字段)
            for field in ("id", "name", "status", "command", "parameters",
                          "start_time", "end_time", "best_round", "best_val",
                          "best_test", "created_by", "created_at",
                          "canonical_config", "git_commit", "environment",
                          "task_dir", "last_round", "exit_code", "error_tail"):
                check(f"4b 响应字段存在: {field}", field in exp)
            check("4b0 配置快照已保存 (canonical_config 含 num_rounds)",
                  bool(exp.get("canonical_config", {}).get("num_rounds")),
                  str(exp.get("canonical_config", {}))[:120])
            cmd = exp.get("command", "")
            check("4c 参数转换为合法 CLI (federated_rounds->num_rounds)",
                  "--num_rounds 1" in cmd, cmd[:200])
            check("4d 参数转换为合法 CLI (learning_rate->lr)",
                  "--lr 0.01" in cmd)
            check("4e 参数转换为合法 CLI (fake_node_count->fake_nodes)",
                  "--fake_nodes 32" in cmd)
            check("4f 参数转换为合法 CLI (knn_k)",
                  "--knn_k 5" in cmd)
            check("4g 布尔 False 正确传 --no-use_weighted_ce",
                  "--no-use_weighted_ce" in cmd, cmd[:200])

            # ---- 状态查询 ----
            print("[5] 任务状态查询")
            st, body = http("GET", "/training/status", token=token)
            check("5a 状态端点可用", st == 200 and body.get("is_running") is True)
            st, body = http("GET", f"/experiments/{exp['id']}", token=token)
            check("5b 实验详情可查询", st == 200 and body.get("id") == exp["id"])

            # ---- 有效任务: 等待完成并校验状态/日志/指标 ----
            print("[6] 有效任务完成链路")
            timeout = 600
            t0 = time.time()
            status = None
            while time.time() - t0 < timeout:
                st, body = http("GET", f"/experiments/{exp['id']}", token=token)
                status = body.get("status") if st == 200 else None
                if status in ("finished", "failed", "stopped"):
                    break
                time.sleep(5)
            check("6a 任务最终状态可查询", status is not None, str(status))
            check("6b 有效配置任务正常完成", status == "finished",
                  f"status={status}")
            st, logs = http("GET", f"/experiments/{exp['id']}/logs",
                            token=token)
            lines = " ".join(l["line"] for l in logs) if st == 200 else ""
            check("6c 日志可读取且含 Core method 声明",
                  st == 200 and len(logs) > 0 and "Core method" in lines,
                  lines[-200:])
            st, metrics = http("GET", f"/experiments/{exp['id']}/metrics",
                               token=token)
            names = [m["metric_name"] for m in metrics] if st == 200 else []
            check("6d 指标已解析入库 (含 global_val/best_val)",
                  st == 200 and "global_val" in names and "best_val" in names,
                  str(names))
            check("6e 最佳轮次/指标写入实验记录",
                  body.get("best_round") is not None and
                  body.get("best_val") is not None, str(body.get("best_round")))

            # ---- B4 完整方法 + 参数热更新 (平台端到端) ----
            print("[6B] B4 完整方法 + 热更新 E2E")
            b4_params = {
                "root": "./dataset", "dataset": "Cora",
                "task_mode": "anomaly_binary",
                "normal_classes": "0,1,2,3", "anomaly_classes": "4,5,6",
                # 轮数给足余量: 热更新在第 1 轮后提交, 需留出"后续轮次"以验证
                # "下一轮生效"语义。原先 3 轮时窗口只有第 2 轮, 训练端在正式轮次前
                # 还要跑联邦扩散预训练(约 3s), 轮次边界会被挤到任务尾声导致更新落空。
                "num_clients": 2, "federated_rounds": 8, "num_epochs": 1,
                "learning_rate": 0.01, "hid_dim": 32, "dropout": 0.2,
                "use_weighted_ce": True,
                "contrastive_mode": "subgraph_cross_view",
                "contrastive_anchor_scope": "all_nodes",
                "contrastive_batch_size": 8, "rwr_subgraph_size": 3,
                "edge_perturb_ratio": 0.2, "contrastive_temperature": 0.5,
                "lambda_subgraph": 0.1,
                "ckr_mode": "static_topology", "distill_weighting": "static_ckr",
                "fake_class_strategy": "balanced", "fake_node_count": 48,
                "knn_k": 5, "generator_steps": 1, "distillation_steps": 1,
                "diffusion_steps": 4, "diffusion_hidden": 32,
                "generator_warmup_rounds": 0, "seed": 2024,
            }
            st, exp4 = http("POST", "/training/start",
                            {"name": "smoke-b4-hotupdate", "parameters": b4_params},
                            token=token)
            check("6B-a B4 任务创建", st == 200, str(st))
            if st == 200:
                # 等待 round 1 开始 (last_round >= 1 由 round_started 事件写入)
                # 保证热更新请求在轮次边界之后提交, 验证"下一轮生效"语义
                t0 = time.time()
                while time.time() - t0 < 300:
                    st, body = http("GET", f"/experiments/{exp4['id']}", token=token)
                    if st == 200 and (body.get("last_round") or 0) >= 1 \
                            and body.get("status") == "running":
                        break
                    time.sleep(2)
                # 提交热更新: learning_rate -> 0.005 (下一轮生效)
                st, upd = http("POST", "/training/update",
                               {"parameter": "learning_rate", "value": 0.005},
                               token=token)
                check("6B-b 热更新请求被接受 (pending)",
                      st == 200 and upd.get("status") == "pending", str(upd))
                st, bad = http("POST", "/training/update",
                               {"parameter": "hid_dim", "value": 128}, token=token)
                check("6B-c 非白名单参数被拒绝 (需要重启)",
                      st == 400 and "重新启动" in str(bad.get("detail")), str(bad))
                # 等待完成
                t0 = time.time()
                while time.time() - t0 < 900:
                    st, body = http("GET", f"/experiments/{exp4['id']}", token=token)
                    if st == 200 and body.get("status") in ("finished", "failed", "stopped"):
                        break
                    time.sleep(5)
                check("6B-d B4 任务完成", body.get("status") == "finished",
                      f"status={body.get('status')}")
                # 热更新审计记录
                st, updates = http("GET", f"/training/updates?experiment_id={exp4['id']}",
                                   token=token)
                applied = [u for u in updates
                           if u.get("status") == "applied"
                           and u.get("parameter") == "learning_rate"] if st == 200 else []
                check("6B-e 热更新已应用且为下一轮生效 (effective_round >= 2)",
                      len(applied) >= 1 and applied[0].get("effective_round") >= 2,
                      str(updates)[:200])
                # 结构化事件指标 (client / ckr / generator / distillation)
                st, metrics = http("GET", f"/experiments/{exp4['id']}/metrics",
                                   token=token)
                names = [m["metric_name"] for m in metrics] if st == 200 else []
                srcs = {m["metric_name"]: m["source"] for m in metrics} if st == 200 else {}
                check("6B-f 客户端指标入库 (ce_loss)",
                      "ce_loss" in names and "total_loss" in names)
                check("6B-g CKR 指标入库 (ckr_c*)",
                      any(n.startswith("ckr_c") for n in names))
                check("6B-h 生成器/蒸馏指标入库",
                      "generator_loss" in names and "distillation_loss" in names
                      and "fake_graph_edges" in names)
                check("6B-i 资源指标入库",
                      "round_time_sec" in names and "generator_peak_gpu_mb" in names)
                # checkpoint 落盘
                ckpt_dir = os.path.join(exp4.get("task_dir", ""), "checkpoints")
                check("6B-j best checkpoint 已保存",
                      os.path.exists(os.path.join(ckpt_dir, "best.pt")),
                      ckpt_dir)

            # ---- 训练异常可被平台读取 (非法配置被 422 拒绝; 运行时失败可读) ----
            print("[7] 失败任务可读性")
            st, bad_body = http("POST", "/training/start",
                                {"name": "smoke-bad-dataset",
                                 "parameters": {"dataset": "Nonexistent",
                                                "num_clients": 2}},
                                token=token)
            check("7a 非法数据集被校验层 422 拒绝",
                  st == 422 and ("not in choices" in str(bad_body) or "unsupported" in str(bad_body)), str(bad_body)[:150])
            st, exp2 = http("POST", "/training/start",
                            {"name": "smoke-fail-test",
                             "parameters": {"dataset": "Cora", "num_clients": 2,
                                            "resume_checkpoint": "/nonexistent/best.pt",
                                            "distill_weighting": "none",
                                            "contrastive_mode": "none",
                                            "federated_rounds": 1, "num_epochs": 1}},
                            token=token)
            check("7b 运行时失败任务可以创建", st == 200, str(st))
            timeout = 300
            t0 = time.time()
            status2 = None
            while time.time() - t0 < timeout:
                st, body2 = http("GET", f"/experiments/{exp2['id']}", token=token)
                status2 = body2.get("status") if st == 200 else None
                if status2 in ("finished", "failed", "stopped"):
                    break
                time.sleep(3)
            check("7c 训练异常被记录为 failed",
                  status2 == "failed", f"status={status2}")
            if status2 == "failed":
                st, logs2 = http("GET", f"/experiments/{exp2['id']}/logs",
                                 token=token)
                lines2 = " ".join(l["line"] for l in logs2) if st == 200 else ""
                check("7d 失败日志可读取且含错误信息",
                      st == 200 and len(logs2) > 0 and
                      any(k in lines2 for k in ("Error", "error", "BLOCKED",
                                                "Traceback", "Exception",
                                                "not found")),
                      lines2[-200:])

        # ---- 日志/指标接口可用性 (对失败任务也返回 200) ----
        print("[7] 日志与指标接口")
        if st == 200 and exp.get("id"):
            eid = exp["id"]
            st, _ = http("GET", f"/experiments/{eid}/logs", token=token)
            check("8a 日志接口可用", st == 200)
            st, _ = http("GET", f"/experiments/{eid}/metrics", token=token)
            check("8b 指标接口可用", st == 200)
            st, body = http("GET", "/experiments/", token=token)
            check("8c 实验列表可用", st == 200 and isinstance(body, list))

        # ---- 停止服务 ----
        print("[8] 停止服务")
        server.terminate()
        server.wait(timeout=10)
        check("9a 服务正常停止", True)

    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except Exception:
                server.kill()

    print(f"\n== 结果: {len(passed)} 通过, {len(failed)} 失败 ==")
    if failed:
        print(f"失败项: {failed}")
        sys.exit(1)
    print("ALL BACKEND SMOKE TESTS PASSED ✓")


if __name__ == "__main__":
    main()
