# FedTAD Training Manager

FedTAD 联邦图学习训练的可视化管理平台。

## 项目结构

```
research-training/
├── backend/          # FastAPI 后端
│   ├── app/
│   │   ├── main.py           # FastAPI 应用入口
│   │   ├── config.py         # 配置（从 .env 读取）
│   │   ├── database.py       # SQLAlchemy 异步引擎
│   │   ├── models.py         # 数据库模型
│   │   ├── schemas.py        # Pydantic 数据模型
│   │   ├── auth.py           # JWT 认证 / bcrypt
│   │   ├── training.py       # 训练子进程管理
│   │   ├── log_parser.py     # 训练日志解析
│   │   ├── websocket_manager.py  # WebSocket 连接管理
│   │   └── routers/
│   │       ├── auth.py        # 登录 API
│   │       ├── experiments.py # 实验记录 API
│   │       └── training.py   # 训练启停 API + WebSocket
│   ├── requirements.txt
│   ├── .env.example
│   └── README.md
├── frontend/          # Vue 3 前端
│   ├── src/
│   │   ├── api/        # Axios API 封装
│   │   ├── router/     # Vue Router
│   │   ├── stores/     # Pinia 状态管理
│   │   ├── views/      # 页面组件
│   │   └── components/ # 可复用组件
│   ├── package.json
│   ├── vite.config.js
│   └── README.md
└── README.md
```

## 启动步骤

### 1. 创建 PostgreSQL 数据库

```bash
createdb fedtad
```

### 2. 配置后端

```bash
cd backend
cp .env.example .env
# 编辑 .env：设置数据库连接和密钥
pip install -r requirements.txt
```

### 3. 启动后端

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. 启动前端

```bash
cd frontend
npm install
npm run dev
```

### 5. 登录

浏览器打开 http://localhost:5173

默认管理员账户：`admin` / `admin123`

**请务必在生产环境修改默认密码！**
