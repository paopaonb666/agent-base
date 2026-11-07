# syntax=docker/dockerfile:1
#
# agent-base（Java 重构版）runtime image。
# 多阶段构建：builder 用 Maven 产出两个 exec jar（Spring Boot fat jar）；
# runtime 只带 JRE 与 jar。Java 侧没有 alembic——表结构由 SchemaMigrations
# 在服务启动时自举（memory 与 core_files 两 store 声明式迁移），所以镜像
# 内不含任何迁移资产，compose 里也不需要迁移命令。
#
# 构建与运行：
#   docker build -t agent-base-java:local ./java
#   docker run --rm -p 8000:8000 --env-file .env agent-base-java:local
# 或经仓库根 docker-compose.yml 的 app-java 服务（profiles: ["java"]）。

FROM maven:3.9-eclipse-temurin-21 AS builder

ENV MAVEN_OPTS="-Dfile.encoding=UTF-8"

WORKDIR /build

# 整个 java/ 目录（父 POM + 两个模块 + 项目级 mvn-settings.xml）。
COPY . .

# -Dmaven.repo.local 覆盖 mvn-settings.xml 里写给开发机（Windows）的
# localRepository——容器内必须落在 Linux 路径上。
# -s 使用项目级 settings：阿里云 https 镜像，不依赖容器内的全局 settings。
RUN mvn -B -s mvn-settings.xml -Dmaven.repo.local=/build/.m2-repo \
        package -DskipTests


FROM eclipse-temurin:21-jre AS runtime

# Dedicated non-root identity (uid/gid 10001)，对齐 Python 镜像。
RUN groupadd --gid 10001 app \
    && useradd --uid 10001 --gid 10001 --no-create-home \
       --shell /usr/sbin/nologin app \
    && apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 两个 exec jar 都带上：完整部署形态（基座+记忆包）是服务入口；基座
# exec jar 仅服务 CLI 用法（java -jar agent-base-0.3.0-exec.jar --message …），
# 服务运行本身不需要它。
COPY --from=builder /build/agent-base/target/agent-base-0.3.0-exec.jar /app/agent-base-0.3.0-exec.jar
COPY --from=builder /build/agent-base-memory/target/agent-base-memory-0.3.0-exec.jar /app/agent-base-memory-0.3.0-exec.jar

# Writable location for the sqlite fallback checkpointer / uploads.
RUN mkdir -p /app/data && chown -R app:app /app

USER app

EXPOSE 8000

# Liveness probe: /health answers HTTP 200 even when degraded.
# Readiness should additionally parse the body and require status == ok.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["curl", "-fsS", "http://127.0.0.1:8000/health"]

ENTRYPOINT ["java", "-jar", "/app/agent-base-memory-0.3.0-exec.jar"]
