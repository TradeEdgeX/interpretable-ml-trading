# VS Code / Cursor 编辑器配置

本目录包含项目级别的编辑器配置，用于统一代码格式化和开发体验。

## 自动格式化设置

### 保存时自动格式化

已配置在保存文件时自动格式化代码：

- **Python**: 使用 `black` 格式化器，行长度 88 字符
- **JSON/YAML**: 自动格式化
- **Markdown**: 自动格式化

### 配置说明

- `.vscode/settings.json`: 编辑器设置
  - `editor.formatOnSave`: 保存时自动格式化
  - `editor.codeActionsOnSave`: 保存时自动整理导入
  - Python 使用 `ms-python.black-formatter` 扩展

- `.editorconfig`: 跨编辑器配置文件
  - 统一缩进、换行符等基础格式
  - 支持多种编辑器（VS Code, Cursor, PyCharm 等）

- `.vscode/extensions.json`: 推荐扩展列表
  - 安装推荐扩展以获得最佳体验

## 使用说明

### 1. 安装推荐扩展

打开命令面板（`Ctrl+Shift+P` / `Cmd+Shift+P`），运行：
```
Extensions: Show Recommended Extensions
```

或直接安装：
- `ms-python.black-formatter` - Black 格式化器
- `editorconfig.editorconfig` - EditorConfig 支持

### 2. 验证配置

1. 打开任意 Python 文件
2. 修改代码（例如添加多余空格）
3. 保存文件（`Ctrl+S` / `Cmd+S`）
4. 代码应该自动格式化

### 3. 手动格式化

如果自动格式化未生效，可以手动格式化：
- 命令面板 → `Format Document` 或 `Shift+Alt+F` (Windows/Linux) / `Shift+Option+F` (Mac)

## 与 Makefile 的一致性

编辑器配置与项目 Makefile 中的格式化命令保持一致：
```bash
make format  # 使用 black 格式化代码
```

两者使用相同的配置（行长度 88 字符）。

## 故障排除

### 格式化不工作？

1. **检查扩展是否安装**:
   - 确保安装了 `ms-python.black-formatter`
   - 确保安装了 `editorconfig.editorconfig`

2. **检查 Python 解释器**:
   - 确保选择了正确的 Python 解释器
   - 命令面板 → `Python: Select Interpreter`

3. **检查 Black 是否安装**:
   ```bash
   python -m black --version
   ```

4. **重新加载窗口**:
   - 命令面板 → `Developer: Reload Window`

### 格式化结果与预期不符？

- 检查 `.vscode/settings.json` 中的 `black-formatter.args`
- 确保与 `Makefile` 中的 black 配置一致

