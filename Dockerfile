FROM public.ecr.aws/lambda/python:3.12

# Keep Python predictable and quiet in Lambda logs.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Lambda filesystem is read-only outside /tmp, so force all runtime state there.
ENV DATA_ROOT_DIRECTORY=/tmp/.cognee/data
ENV SYSTEM_ROOT_DIRECTORY=/tmp/.cognee/system
ENV CACHE_ROOT_DIRECTORY=/tmp/.cognee/cache

WORKDIR ${LAMBDA_TASK_ROOT}

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir \
    "cognee>=1.2.2" \
    "fastapi>=0.139.0" \
    "mysql-connector-python>=9.7.0" \
    "pdfplumber>=0.11.10" \
    "pypdf>=6.14.2" \
    "python-dotenv>=1.0.1" \
    "python-multipart>=0.0.32" \
    "rapidocr-onnxruntime>=1.4.4" \
    "uvicorn[standard]>=0.50.0" \
    "mangum>=0.19.0"

COPY . ${LAMBDA_TASK_ROOT}

# Registry uses PROJECT_ROOT/patients.db; map that path to writable /tmp.
RUN ln -sf /tmp/patients.db ${LAMBDA_TASK_ROOT}/patients.db

CMD ["lambda_handler.handler"]