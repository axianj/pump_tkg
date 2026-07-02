# 离心泵故障诊断时序知识图谱平台
# 基于 Python 3.12 slim 镜像
FROM python:3.12-slim

LABEL org.opencontainers.image.title="pump_tkg"
LABEL org.opencontainers.image.description="离心泵故障诊断时序知识图谱平台"
LABEL org.opencontainers.image.version="0.2.0"

# 环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖 — 分层安装以利用 Docker 缓存
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install kuzu && \
    pip install python-docx PyPDF2

# 应用代码
COPY . .

# 运行时需要的目录
RUN mkdir -p /app/data/output /app/data/knowledge

# 暴露 Web 端口
EXPOSE 8501

# 默认启动 Streamlit
CMD ["streamlit", "run", "web/app.py"]
