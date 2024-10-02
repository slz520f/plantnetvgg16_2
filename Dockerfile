# ビルドステージ
FROM python:3.11-slim-buster AS builder

# 環境変数の設定
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 必要なライブラリのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# 依存関係ファイルをコピーしてインストール
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt 

# アプリケーションコードをコピー
COPY . .

# 実行ステージ
FROM python:3.11-slim-buster

# 環境変数の設定
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT 5000

# 必要なライブラリのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# ビルドステージから必要なファイルのみをコピー
COPY --from=builder /app /app

# アプリケーションの実行
CMD ["sh", "-c", "gunicorn -b 0.0.0.0:$PORT plantnet_clone_1.wsgi:application"]
