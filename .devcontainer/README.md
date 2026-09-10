# Dev Container 配置说明

## 文件权限和用户配置

### 当前配置

- **容器用户**: `dev` (UID 1000, GID 1000)
- **remoteUser**: `dev`
- **文件权限**: 通过 UID/GID 匹配，而不是用户名

### 为什么当前配置是安全的？

Linux 文件系统使用 **UID/GID** 来标识文件所有者，而不是用户名。这意味着：

- 如果您的 host 系统用户 UID 也是 1000（大多数 Linux 系统的默认值），文件权限会自动匹配
- 容器内 `dev` 用户创建的文件，在 host 系统上会被识别为您的主用户（UID 1000）
- 即使 host 系统用户名不同（如 `ubuntu`、`yin`、`user` 等），只要 UID 是 1000，权限就正确
- **使用 `dev` 这个通用名称更合适**，因为它不绑定到特定的个人用户名，更加通用和匿名

### 如何检查您的 host UID？

在 host 系统上运行：
```bash
id -u  # 查看当前用户 UID
id -g  # 查看当前用户 GID
```

如果输出是 `1000`，当前配置即可正常工作。

### 如果您的 host UID 不是 1000

有两种方案：

#### 方案 1：修改 Dockerfile（推荐）

在构建镜像时传递 build args：

1. 修改 `.devcontainer/devcontainer.json`，添加 `build` 配置：
```json
{
  "build": {
    "dockerfile": "Dockerfile",
    "args": {
      "USERNAME": "dev",
      "USER_UID": "${localEnv:UID}",
      "USER_GID": "${localEnv:GID}"
    }
  }
}
```

2. 在 host 系统上设置环境变量：
```bash
export UID=$(id -u)
export GID=$(id -g)
```

#### 方案 2：使用 devcontainer features

使用 Microsoft 的 devcontainer features 来自动创建匹配的用户（需要修改配置，较复杂）。

## mlbot 命令配置

`mlbot` 命令通过 `postCreateCommand` 自动配置：

1. 安装包：`pip install -e .`
2. 配置 PATH：添加 `~/.local/bin` 到 fish 和 bash 的 PATH
3. 每次启动 devcontainer 时，`mlbot` 命令自动可用

配置位置：
- Fish: `~/.config/fish/config.fish`
- Bash: `~/.bashrc`

## SSH 配置

SSH 密钥从 host 系统自动复制到容器：
- Host SSH 路径：`${HOME}/.ssh`（或 Windows: `${USERPROFILE}/.ssh`）
- 容器内路径：`$HOME/.ssh`
- 自动配置 SSH agent forwarding

