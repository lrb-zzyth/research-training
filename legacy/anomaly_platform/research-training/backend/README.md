# FedTAD Training Manager — Backend

## 环境要求

- Python 3.9+
- PostgreSQL 14+

## 快速开始

### 1. 创建 PostgreSQL 数据库

由于默认的 peer authentication 限制，**不要直接使用** `psql -U postgres`。

推荐使用系统用户 postgres 来管理数据库：

```bash
# 进入 postgres 管理 shell
sudo -u postgres psql

# 在 psql 中创建数据库
CREATE DATABASE fedtad;

# 创建普通数据库用户（用于项目连接）
CREATE USER fedtad_user WITH PASSWORD 'fedtad_password';

# 授予数据库访问权限
GRANT ALL PRIVILEGES ON DATABASE fedtad TO fedtad_user;

# 退出
\q
```

如果已有数据库但需要重建，可以先删除再创建：

```bash
sudo -u postgres psql -c "DROP DATABASE IF EXISTS fedtad;"
sudo -u postgres psql -c "CREATE DATABASE fedtad;"
```

> **注意**：`fedtad_password` 仅为本地开发密码，生产环境必须修改。

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 中的数据库连接信息
```

开发环境推荐配置：

```
DATABASE_URL=postgresql+asyncpg://fedtad_user:fedtad_password@127.0.0.1:5432/fedtad
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 启动后端

从 `research-training/backend/` 目录运行：

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API 文档自动生成在 http://localhost:8000/docs。

### 5. 默认用户与注册

首次启动自动创建默认管理员账户：

- 用户名：`admin`
- 密码：`admin123`

**生产环境请务必修改默认密码！**

普通用户可以通过注册页面创建新账户：

- 访问前端 `/register` 页面
- 填写用户名、密码、确认密码
- 注册成功后跳转到登录页
- 使用新账户登录后进入训练管理界面

## 数据库维护

### 表结构

应用启动时会自动调用 `Base.metadata.create_all()` 创建所有表，无需手动执行迁移脚本。

| 表名 | 说明 |
|---|---|
| `users` | 用户账户（username 唯一索引） |
| `experiments` | 训练实验记录 |
| `training_logs` | 训练日志行 |
| `training_metrics` | 训练指标（按 round + source） |

### 清空数据库

如果需要重建（会丢失所有数据）：

```bash
sudo -u postgres psql -c "DROP DATABASE IF EXISTS fedtad;"
sudo -u postgres psql -c "CREATE DATABASE fedtad;"
```

## 安全提醒

- `.env` 文件包含数据库密码和 JWT 密钥，**不要提交到 GitHub**
- `.env.example` 中的密码仅为本地开发使用
- 生产环境请更换 `SECRET_KEY` 并设置强数据库密码
