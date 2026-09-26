# FedTAD Training Manager — Frontend

## 环境要求

- Node.js 18+
- npm 9+

## 快速开始

### 1. 安装依赖

```bash
npm install
```

### 2. 启动开发服务器

```bash
npm run dev
```

前端默认运行在 http://localhost:5173。

### 3. 构建生产版本

```bash
npm run build
```

构建产物在 `dist/` 目录。

## 页面

| 路径 | 说明 |
|---|---|
| `/login` | 用户登录 |
| `/register` | 用户注册 |
| `/training/config` | 训练参数配置 |
| `/training/monitor/:id` | 训练监控（实时日志 + 指标曲线） |
| `/experiments` | 实验历史记录 |

注册成功后跳转到登录页，使用新账户登录即可进入训练管理界面。

## 配置

Vite 开发服务器已代理 `/api` 请求到后端 `http://localhost:8000`，确保后端先启动。

如需修改代理目标，编辑 `vite.config.js`。
