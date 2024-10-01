# ビルドステージ
FROM python:3.11-slim-buster AS builder

# 環境変数の設定

ENV DATABASE_URL='postgres://u1stg7p15lfd4d:p322414979cc693b07786d189e4a6c135e7e89f6f32491451f68ac4da56b5f3a7@cc0gj7hsrh0ht8.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d3sh29do69ja1d'

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

# 静的ファイルの収集
# RUN python manage.py collectstatic --noinput

# 実行ステージ
FROM python:3.11-slim-buster

# 環境変数の設定
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 必要なライブラリのインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /app

# ビルドステージから必要なファイルのみをコピー
COPY --from=builder /app /app

# アプリケーションの実行コマンド
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "plantnet_clone_1.wsgi:application"]
