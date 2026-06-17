.PHONY: help install dev-install test demo clean build upload lint format type-check

# 默认目标
help:
	@echo "AI验证系统 - 可用命令:"
	@echo "  install      - 安装项目依赖"
	@echo "  dev-install  - 安装开发依赖"
	@echo "  demo         - 运行演示脚本"
	@echo "  test         - 运行测试"
	@echo "  lint         - 代码检查"
	@echo "  format       - 代码格式化"
	@echo "  type-check   - 类型检查"
	@echo "  clean        - 清理构建文件"
	@echo "  build        - 构建项目"
	@echo "  upload       - 上传到PyPI"

# 安装依赖
install:
	pip install -r requirements.txt

# 安装开发依赖
dev-install: install
	pip install -e ".[dev]"

# 运行演示
demo:
	python scripts/demo.py

# 运行测试
test:
	python -m pytest tests/ -v

# 代码检查
lint:
	flake8 src/ scripts/ --max-line-length=100
	mypy src/

# 代码格式化
format:
	black src/ scripts/ --line-length=100

# 类型检查
type-check:
	mypy src/ai_verifier/

# 清理构建文件
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# 构建项目
build: clean
	python -m build

# 上传到PyPI
upload: build
	python -m twine upload dist/*

# 项目设置
setup: dev-install
	pre-commit install

# 全面检查
check: format lint type-check test
	@echo "✅ 所有检查通过！" 